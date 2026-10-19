#!/usr/bin/env python3
"""Reuse the trained Phase 3 policy to separate numerical, timing and task parity.

Actual CPU training/rollouts; synthetic cart; imposed latencies; no VLA checkpoint,
integer inference kernel, measured speedup, middleware or physical robot.
"""
import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import statistics
import sys

ROOT = Path(__file__).resolve().parents[4]
FOUNDATIONS = ROOT / 'Phase 3 - Artificial Intelligence/0. AI and Robot Learning Foundations/labs'
if not (FOUNDATIONS / 'policy_lab.py').is_file():
    raise ImportError('Keep this lab in the full course checkout; Phase 3 policy labs are required')
sys.path.insert(0, str(FOUNDATIONS))
import policy_lab
import timed_policy_lab as timing_lab


@dataclass(frozen=True)
class ActionContract:
    features: tuple = ('goal_minus_position_m', 'velocity_m_s')
    action_kind: str = 'acceleration'
    action_unit: str = 'm/s^2'
    frame: str = 'cart_line_positive_right'
    timestamp_kind: str = 'capture_ms_in_sim_clock'
    limit: float = 1.0


CONTRACT = ActionContract()
BUDGETS = {'raw_action_mse_m2_s4': 1e-5, 'raw_action_max_m_s2': 0.01,
           'max_applied_age_ms': 250, 'max_position_divergence_m': 0.02,
           'allowed_lost_fixture_episodes': 0, 'population_noninferiority_margin': 0.02,
           'confidence_alpha': 0.05}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def make_tape(rows, policy):
    counters = {}
    tape = []
    for row in rows:
        frame = counters.get(row.episode, 0)
        counters[row.episode] = frame + 1
        observation = {'episode': row.episode, 'frame': frame,
                       'capture_ms': frame * 100, 'error_m': row.error,
                       'velocity_m_s': row.velocity}
        tape.append({'episode': row.episode, 'frame': frame,
                     'observation_sha256': digest(observation),
                     'action': policy(row.error, row.velocity)})
    return tape


def replay_metrics(reference, candidate, reference_contract=CONTRACT,
                   candidate_contract=CONTRACT):
    if reference_contract != candidate_contract:
        raise ValueError('Action contract differs: compare semantics before numbers')
    if not math.isfinite(reference_contract.limit) or reference_contract.limit <= 0:
        raise ValueError('Action limit must be positive and finite')
    def index(records):
        if not records:
            raise ValueError('Empty replay cannot establish parity')
        result = {}
        for row in records:
            if type(row['episode']) is not int or type(row['frame']) is not int or min(row['episode'], row['frame']) < 0:
                raise ValueError('Episode/frame keys must be nonnegative integers')
            key = (row['episode'], row['frame'])
            if key in result:
                raise ValueError('Duplicate replay key')
            if type(row['action']) not in (float, int) or not math.isfinite(row['action']):
                raise ValueError('Action must be a finite scalar acceleration')
            if not isinstance(row['observation_sha256'], str) or len(row['observation_sha256']) != 64:
                raise ValueError('Missing observation identity')
            result[key] = row
        return result
    refs, cands = index(reference), index(candidate)
    if refs.keys() != cands.keys():
        raise ValueError('Missing or extra replay frames')
    differences, clipped_differences = [], []
    clip = lambda value: max(-reference_contract.limit, min(reference_contract.limit, value))
    for key in sorted(refs):
        ref, cand = refs[key], cands[key]
        if ref['observation_sha256'] != cand['observation_sha256']:
            raise ValueError('Replay observation identity mismatch')
        differences.append(cand['action'] - ref['action'])
        clipped_differences.append(clip(cand['action']) - clip(ref['action']))
    if not all(math.isfinite(value) and math.isfinite(value * value) for value in differences):
        raise ValueError('Action difference overflow')
    return {'frames': len(differences),
            'raw_mse_m2_s4': statistics.fmean(x*x for x in differences),
            'raw_max_error_m_s2': max(abs(x) for x in differences),
            'signed_mean_error_m_s2': statistics.fmean(differences),
            'clipped_mse_m2_s4': statistics.fmean(x*x for x in clipped_differences)}


def binomial_upper(k, n, alpha=0.05):
    """Exact one-sided Clopper-Pearson upper bound, numerical CDF inversion."""
    if type(n) is not int or type(k) is not int or not 0 <= k <= n or n < 1:
        raise ValueError('Need integer counts 0 <= k <= n, n > 0')
    if not math.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError('alpha must be between zero and one')
    if k == n:
        return 1.0
    if k == 0:
        return -math.expm1(math.log(alpha) / n)
    def cdf(p):
        logs = [math.lgamma(n+1)-math.lgamma(j+1)-math.lgamma(n-j+1)
                + j*math.log(p)+(n-j)*math.log1p(-p) for j in range(k+1)]
        largest = max(logs)
        return math.exp(largest) * math.fsum(math.exp(x-largest) for x in logs)
    low, high = 0.0, 1.0
    for _ in range(70):
        mid = (low + high) / 2
        if mid == low or mid == high:
            break
        if cdf(mid) > alpha:
            low = mid
        else:
            high = mid
    return high


def paired_outcomes(reference, candidate, margin=0.02, alpha=0.05):
    """Conservative noninferiority: delta=P(help)-P(harm) >= -P(harm)."""
    if not math.isfinite(margin) or not 0 <= margin < 1:
        raise ValueError('margin must lie in [0,1)')
    if not reference or reference.keys() != candidate.keys():
        raise ValueError('Require the same nonempty episode identifiers')
    if any(type(value) is not bool for values in (reference, candidate) for value in values.values()):
        raise ValueError('Success fields must be booleans')
    n = len(reference)
    harmed = sum(reference[key] and not candidate[key] for key in reference)
    helped = sum(candidate[key] and not reference[key] for key in reference)
    upper = binomial_upper(harmed, n, alpha)
    return {'episodes': n, 'harmed': harmed, 'helped': helped,
            'observed_success_difference': (helped-harmed)/n,
            'harm_probability_upper': upper, 'conservative_difference_lower': -upper,
            'alpha': alpha, 'margin': margin,
            'population_gate': 'PASS' if upper <= margin else 'INCONCLUSIVE',
            'assumption': 'Independent paired episodes from the stated fixed distribution; no multiple-comparison correction'}


def fold_normalization(weights, scaler):
    means, scales = scaler
    if len(weights) != 3 or len(means) != 2 or len(scales) != 2:
        raise ValueError('This fixture requires two features and a bias')
    if not all(math.isfinite(x) for x in (*weights, *means, *scales)) or any(x <= 0 for x in scales):
        raise ValueError('Nonfinite model or invalid normalization scale')
    physical = [weights[j] / scales[j] for j in range(2)]
    return physical + [weights[2] - sum(physical[j]*means[j] for j in range(2))]


def quantized_weights(weights):
    scale = max(abs(x) for x in weights) / 127
    if scale == 0:
        return list(weights), {'scale': 1.0, 'integers': [0]*len(weights)}
    ints = [max(-127, min(127, round(x/scale))) for x in weights]
    return [x*scale for x in ints], {'scale': scale, 'integers': ints}


def run(seed=7, episodes=12):
    if type(episodes) is not int or not 1 <= episodes <= 100:
        raise ValueError('episodes must be 1 through 100 for this bounded lab')
    trained = policy_lab.run(seed=seed)
    weights = trained['weights_normalized_inputs']
    scaler = (trained['scaler_training_mean'], trained['scaler_training_std'])
    reference = lambda e, v: policy_lab.predict(weights, e, v, scaler)
    folded = fold_normalization(weights, scaler)
    quantized, quant_metadata = quantized_weights(weights)
    variants = [
        ('identity', reference, timing_lab.Timing('fast_latest', 10, transport_ms=0)),
        ('folded_normalization', lambda e,v: folded[0]*e+folded[1]*v+folded[2], timing_lab.Timing('fast_latest', 10, transport_ms=0)),
        ('int8_weight_arithmetic', lambda e,v: policy_lab.predict(quantized,e,v,scaler), timing_lab.Timing('fast_latest', 10, transport_ms=0)),
        ('missing_normalization_fault', lambda e,v: weights[0]*e+weights[1]*v+weights[2], timing_lab.Timing('fast_latest', 10, transport_ms=0)),
        ('same_weights_slow_fifo', reference, timing_lab.Timing('slow_fifo_unchecked', 120, queue_mode='fifo', check_age=False)),
        ('same_weights_slow_latest', reference, timing_lab.Timing('slow_latest_checked', 120)),
        ('same_weights_over_budget', reference, timing_lab.Timing('over_budget', 350)),
    ]
    test_rows = policy_lab.split_episodes(policy_lab.make_dataset(seed))[2]
    ref_tape = make_tape(test_rows, reference)
    rng = random.Random(501)
    initials = [(rng.uniform(-1.4,1.4),rng.uniform(-0.3,0.3)) for _ in range(episodes)]
    ref_runs = [timing_lab.simulate(reference,variants[0][2],initial,i,True) for i,initial in enumerate(initials)]
    ref_outcomes = {row['session']:row['success'] for row in ref_runs}
    results = []
    for name, policy, timing in variants:
        numerics = replay_metrics(ref_tape, make_tape(test_rows, policy))
        rows = [timing_lab.simulate(policy,timing,initial,i,True) for i,initial in enumerate(initials)]
        divergences=[]
        for ref, cand in zip(ref_runs, rows):
            if len(ref['trace']) != len(cand['trace']):
                raise ValueError('Trajectory horizons differ')
            differences=[]
            for left,right in zip(ref['trace'],cand['trace']):
                if left['time_ms'] != right['time_ms']:
                    raise ValueError('Trajectory timestamps differ')
                differences.append(abs(left['position_m']-right['position_m']))
            divergences.append(max(differences))
        outcomes = paired_outcomes(ref_outcomes,{row['session']:row['success'] for row in rows},
                                  BUDGETS['population_noninferiority_margin'],BUDGETS['confidence_alpha'])
        max_age=max((row['max_applied_age_ms'] or 0) for row in rows)
        applied=sum(row['applied_policy_intervals'] for row in rows)
        failures=[]
        if numerics['raw_mse_m2_s4'] > BUDGETS['raw_action_mse_m2_s4'] or numerics['raw_max_error_m_s2'] > BUDGETS['raw_action_max_m_s2']:
            failures.append('numerical_action_error')
        if max_age > BUDGETS['max_applied_age_ms']:
            failures.append('expired_observation_applied')
        if applied == 0:
            failures.append('no_policy_action_applied')
        if outcomes['harmed'] > BUDGETS['allowed_lost_fixture_episodes']:
            failures.append('lost_fixed_suite_success')
        if max(divergences) > BUDGETS['max_position_divergence_m']:
            failures.append('trajectory_divergence')
        for row, drift in zip(rows,divergences):
            del row['trace']
            row['max_position_difference_from_reference_m'] = drift
        results.append({'candidate':name,'timing':asdict(timing),'numerics':numerics,
                        'successes':sum(row['success'] for row in rows),'paired_outcomes':outcomes,
                        'max_applied_age_ms':max_age if applied else None,
                        'applied_policy_intervals':applied,'max_position_divergence_m':max(divergences),
                        'fixed_fixture_gate':'FAIL' if failures else 'PASS',
                        'failure_reasons':failures,'per_episode':rows})
    return {'execution_status':'actual CPU training, prediction and cart simulation; timing parameters imposed',
            'python':platform.python_version(),'dataset_seed':seed,'rollout_seed':501,
            'training_dataset_sha256':trained['dataset_sha256'],'split_episode_ids':trained['split_episode_ids'],
            'reference_weights':weights,'reference_scaler':scaler,'folded_weights':folded,
            'int8_arithmetic':quant_metadata,'action_contract':asdict(CONTRACT),'budgets':BUDGETS,
            'reference_tape_sha256':digest(ref_tape),'reference_tape_frames':len(ref_tape),
            'held_out_teacher_rmse_m_s2':trained['test']['pooled_action_rmse_m_s2'],
            'reference_successes':sum(ref_outcomes.values()),'candidates':results,
            'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in [Path(__file__),FOUNDATIONS/'policy_lab.py',FOUNDATIONS/'timed_policy_lab.py']},
            'limitations':['No pretrained VLA, image/language encoder, simulator package, GPU or physical robot.',
                'INT8 values are dequantized into Python floating point; no integer kernel or measured speedup.',
                'Fixed-suite gate is a regression result, not population noninferiority or deployment permission.',
                'The independent current velocity encoder fallback is inherited from the timed-policy fixture.',
                'Multiple candidate population claims would need selection-aware fresh evaluation or correction.']}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed',type=int,default=7)
    parser.add_argument('--episodes',type=int,default=12)
    parser.add_argument('--output',type=Path,default=Path('action-parity-report.json'))
    parser.add_argument('--gate-candidate',help='Return 2 when this named candidate fails the fixed-fixture gate')
    args=parser.parse_args()
    if not 1 <= args.episodes <= 100:
        parser.error('--episodes must be 1 through 100')
    report=run(args.seed,args.episodes)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    for row in report['candidates']:
        print(f"{row['candidate']}: fixture={row['fixed_fixture_gate']}; successes={row['successes']}/{args.episodes}; "
              f"raw action max={row['numerics']['raw_max_error_m_s2']:.6g}; "
              f"population={row['paired_outcomes']['population_gate']}")
    print(f'Saved {args.output}')
    if args.gate_candidate:
        rows=[row for row in report['candidates'] if row['candidate']==args.gate_candidate]
        if not rows:
            parser.error('unknown candidate; use a name printed in the report')
        return 0 if rows[0]['fixed_fixture_gate']=='PASS' else 2
    return 0


if __name__=='__main__':
    raise SystemExit(main())
