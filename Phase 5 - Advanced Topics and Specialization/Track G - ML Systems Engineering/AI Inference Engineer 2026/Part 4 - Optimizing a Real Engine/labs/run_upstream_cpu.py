#!/usr/bin/env python3
"""Build pinned upstream CPU components directly; no weights, CUDA or downloads.

This executes trusted source at the exact reviewed revision. It is not a sandbox
for arbitrary repositories and does not launch the upstream evaluator or server.
"""
import argparse
import hashlib
import json
import platform
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

REVISION = '7689cc77e0e18d4b4f1937cf2a189dcd5a8d90f3'
CASES = {
    'tp_shard': ['tp/shard.cpp'],
    'tp_backend': ['tp/backend_select.cpp'],
    'tp_weight_plan': ['tp/weight_plan.cpp', 'tp/shard.cpp'],
    'tp_weight_residency': ['tp/weight_residency.cpp', 'tp/weight_plan.cpp', 'tp/shard.cpp', 'gguf.cpp'],
    'kimi_k3_config': ['gguf.cpp'],
    'kimi_k3_manifest': ['gguf.cpp'],
    'kimi_k3_decode_plan': ['models/kimi_k3_decode_plan.cpp', 'tp/weight_plan.cpp', 'tp/shard.cpp', 'gguf.cpp'],
}


def run(command, cwd):
    process = subprocess.run(command, cwd=cwd, capture_output=True, text=True, timeout=300)
    return {'command': command, 'exit_code': process.returncode,
            'stdout': process.stdout, 'stderr': process.stderr}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--cxx', default='g++')
    args = parser.parse_args()
    repo = args.repo.resolve()
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip()
    if head != REVISION:
        parser.error(f'expected {REVISION}, got {head}')
    if subprocess.run(['git', 'diff', '--quiet', 'HEAD', '--'], cwd=repo).returncode:
        parser.error('tracked source differs from the reviewed commit; use a clean checkout')
    report = {'reviewed_at_utc': datetime.now(timezone.utc).isoformat(),
              'upstream_url': 'https://github.com/gittensor-ai-lab/sparkinfer-k3',
              'revision': head, 'python': sys.version, 'platform': platform.platform(),
              'compiler': subprocess.check_output([args.cxx, '--version'], text=True).splitlines()[0],
              'scope': 'CPU source/contracts only; no GPU execution, model weights or throughput validation',
              'checks': [], 'upstream_source_sha256': {}}
    # Hash all tracked source used by the exercise and the runtime files reviewed.
    paths = subprocess.check_output(['git', 'ls-files'], cwd=repo, text=True).splitlines()
    for name in paths:
        p = repo/name
        if p.is_file() and (name.startswith(('runtime/', 'bench/scripts/', 'eval/')) or
                            name == 'kernels/csrc/cuda/kimi_k3/k3_kernels.cu'):
            report['upstream_source_sha256'][name] = hashlib.sha256(p.read_bytes()).hexdigest()
    with tempfile.TemporaryDirectory(prefix='course-sparkinfer-cpu-') as temporary:
        build = Path(temporary)
        for name, sources in CASES.items():
            binary = build/name
            command = [args.cxx, '-std=c++17', '-O2', '-I'+str(repo/'runtime/include'),
                       '-I'+str(repo/'include'), str(repo/f'runtime/tests/{name}_cpu_test.cpp'),
                       *[str(repo/'runtime/src'/s) for s in sources], '-o', str(binary)]
            compilation = run(command, build)
            report['checks'].append({'name': name, 'stage': 'compile', **compilation})
            if compilation['exit_code'] == 0:
                execution = run([str(binary)], build)
                report['checks'].append({'name': name, 'stage': 'execute', **execution})
                print(name, execution['exit_code'], flush=True)
            else:
                print(name, 'COMPILE FAILED', compilation['stderr'], flush=True)
        for script in ('test_kimi_k3_baseline.py', 'test_label.py'):
            result = run([sys.executable, str(repo/'bench/scripts'/script)], repo)
            report['checks'].append({'name': script, 'stage': 'python-suite', **result})
            print(script, result['exit_code'], flush=True)
    report['passed'] = all(c['exit_code'] == 0 for c in report['checks'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
