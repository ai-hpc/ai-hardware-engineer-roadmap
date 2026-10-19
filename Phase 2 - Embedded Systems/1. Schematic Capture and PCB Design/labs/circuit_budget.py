"""First-order, lumped I2C teaching model; no hardware or SPICE claim."""
from __future__ import annotations
import argparse
import csv
from dataclasses import dataclass, replace
import json
import math
from pathlib import Path


def positive(value: float, label: str, *, zero: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{label} must be finite and numeric")
    if value < 0 or (value == 0 and not zero):
        raise ValueError(f"{label} must be {'nonnegative' if zero else 'positive'}")
    return float(value)


@dataclass(frozen=True)
class Bus:
    pullups_ohm: tuple[float, ...] = (2200.0,)
    resistor_tolerance: float = 0.05
    rail_min_v: float = 3.135
    rail_max_v: float = 3.465
    capacitance_min_pf: float = 80.0
    capacitance_max_pf: float = 120.0
    leakage_sink_ua: float = 0.0
    low_max_v: float = 0.4
    sink_guarantee_ma: float = 3.0
    high_required_v: float = 2.31
    rise_limit_ns: float = 300.0

    def __post_init__(self):
        if not self.pullups_ohm:
            raise ValueError('at least one pull-up is required')
        for r in self.pullups_ohm:
            positive(r, 'resistor')
        for key, value in vars(self).items():
            if key != 'pullups_ohm':
                positive(value, key, zero=key in {'resistor_tolerance', 'leakage_sink_ua', 'low_max_v'})
        if self.resistor_tolerance >= 1:
            raise ValueError('resistor tolerance must be less than 1')
        if self.rail_min_v > self.rail_max_v or self.capacitance_min_pf > self.capacitance_max_pf:
            raise ValueError('minimum exceeds maximum')
        if self.low_max_v >= self.rail_min_v:
            raise ValueError('low threshold must be below the minimum rail')


def parallel(values: tuple[float, ...]) -> float:
    if not values:
        raise ValueError('empty parallel network')
    return 1 / sum(1 / positive(v, 'resistor') for v in values)


def rise_time(r_ohm: float, c_f: float, rail_v: float, leakage_a: float = 0) -> float:
    """30%-70% of rail; constant sink leakage lowers asymptotic voltage."""
    r = positive(r_ohm, 'resistance')
    c = positive(c_f, 'capacitance')
    rail = positive(rail_v, 'rail')
    leakage = positive(leakage_a, 'leakage', zero=True)
    final = rail - leakage * r
    if final <= 0.7 * rail:
        return math.inf
    return r * c * math.log((final - 0.3 * rail) / (final - 0.7 * rail))


def assess(bus: Bus) -> dict:
    nominal = parallel(bus.pullups_ohm)
    r_min = nominal * (1 - bus.resistor_tolerance)
    r_max = nominal * (1 + bus.resistor_tolerance)
    # All pull-ups share one rail; independent bounded tolerances. These
    # monotonic corners enclose the complete model, not a random sample.
    rise = rise_time(r_max, bus.capacitance_max_pf * 1e-12, bus.rail_min_v, bus.leakage_sink_ua * 1e-6)
    sink_ma = (bus.rail_max_v - bus.low_max_v) / r_min * 1000
    high_v = bus.rail_min_v - bus.leakage_sink_ua * 1e-6 * r_max
    nominal_rise = rise_time(nominal, (bus.capacitance_min_pf + bus.capacitance_max_pf) / 2 * 1e-12, (bus.rail_min_v + bus.rail_max_v) / 2)
    checks = {'rise': rise * 1e9 <= bus.rise_limit_ns,
              'sink_guarantee': sink_ma <= bus.sink_guarantee_ma,
              'high_level': high_v >= bus.high_required_v}
    return {'nominal_parallel_ohm': nominal, 'resistance_range_ohm': [r_min, r_max],
            'nominal_rise_ns_no_leakage': nominal_rise * 1e9,
            'worst_rise_ns': rise * 1e9 if math.isfinite(rise) else None,
            'reaches_70_percent': math.isfinite(rise),
            'worst_current_at_low_max_ma': sink_ma, 'worst_high_v': high_v,
            'checks': checks, 'passes_model': all(checks.values())}


def waveform(r_ohm: float, c_f: float, rail_v: float, step_s: float, leakage_a: float = 0) -> list[tuple[float, float]]:
    """Forward-Euler numerical RC integration from 0 V to 5 time constants."""
    tau = positive(r_ohm, 'resistance') * positive(c_f, 'capacitance')
    positive(rail_v, 'rail'); positive(leakage_a, 'leakage', zero=True)
    positive(step_s, 'time step')
    if not tau / 100000 <= step_s <= tau / 4:
        raise ValueError('choose tau/100000 <= step <= tau/4')
    final = rail_v - leakage_a * r_ohm
    if final <= 0:
        raise ValueError('constant-current model has no positive equilibrium')
    values = [(0.0, 0.0)]
    voltage = 0.0
    for i in range(1, math.ceil(5 * tau / step_s) + 1):
        voltage += step_s * ((rail_v - voltage) / r_ohm - leakage_a) / c_f
        values.append((i * step_s, voltage))
    return values


def crossing(values: list[tuple[float, float]], threshold: float) -> float:
    for (t0, v0), (t1, v1) in zip(values, values[1:]):
        if v0 <= threshold <= v1 and v1 > v0:
            return t0 + (threshold - v0) / (v1 - v0) * (t1 - t0)
    raise ValueError('waveform did not cross threshold')


def scenarios() -> dict[str, Bus]:
    baseline = Bus()
    return {'baseline': baseline,
            'nominal_only': replace(baseline, pullups_ohm=(3300.0,)),
            'four_breakouts': replace(baseline, pullups_ohm=(2200.0,) * 4),
            'long_cable': replace(baseline, capacitance_min_pf=160, capacitance_max_pf=240),
            'leakage_fault': replace(baseline, pullups_ohm=(3300.0,), leakage_sink_ua=300)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scenario', choices=scenarios(), default='baseline')
    parser.add_argument('--probe-pf', type=float, default=0)
    parser.add_argument('--waveform-csv', type=Path)
    args = parser.parse_args()
    positive(args.probe_pf, 'probe capacitance', zero=True)
    bus = scenarios()[args.scenario]
    bus = replace(bus, capacitance_min_pf=bus.capacitance_min_pf + args.probe_pf,
                  capacitance_max_pf=bus.capacitance_max_pf + args.probe_pf)
    result = {'scenario': args.scenario, 'assumptions': vars(bus), 'result': assess(bus)}
    if args.waveform_csv:
        r = parallel(bus.pullups_ohm) * (1 + bus.resistor_tolerance)
        c = bus.capacitance_max_pf * 1e-12
        values = waveform(r, c, bus.rail_min_v, r*c/1000, bus.leakage_sink_ua*1e-6)
        with args.waveform_csv.open('w', newline='') as f:
            writer = csv.writer(f); writer.writerow(['time_s', 'voltage_v']); writer.writerows(values)
        try:
            result['numerical_rise_ns'] = (crossing(values, .7*bus.rail_min_v) - crossing(values, .3*bus.rail_min_v))*1e9
        except ValueError:
            result['numerical_rise_ns'] = None
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
