"""Measure native CPU components with a generated fixture and record context."""
import argparse
import hashlib
import json
import platform
from pathlib import Path
import statistics
import subprocess
import tempfile

from make_fixture import make_fixture


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('executable', type=Path)
    args = parser.parse_args()
    executable = args.executable.resolve(strict=True)
    fixture = make_fixture()
    with tempfile.TemporaryDirectory(prefix='course-gguf-bench-') as directory:
        path = Path(directory) / 'toy.gguf'
        path.write_bytes(fixture)
        completed = subprocess.run([str(executable), str(path)], check=True,
                                   capture_output=True, text=True, timeout=120)
    report = json.loads(completed.stdout)
    report.update(host=platform.platform(), python=platform.python_version(),
                  fixture_sha256=hashlib.sha256(fixture).hexdigest(),
                  executable_sha256=hashlib.sha256(executable.read_bytes()).hexdigest(),
                  timing_note='Seven batch means; cached tiny file; shared host; not tail latency or model tokens/s.')
    for component in ('parse', 'sample'):
        samples = report[f'{component}_us_per_operation']
        report[f'{component}_median_batch_mean_us'] = statistics.median(samples)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
