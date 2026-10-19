"""Opt-in native checks of twelve actual Guide.md snippets plus six contracts.

Requires C++17 and oneTBB development files; installs nothing. Bodies with
application inputs receive a small harness and independent expected answers.
"""
from pathlib import Path
import argparse
import datetime
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--compiler', default='g++')
parser.add_argument('--include-dir', type=Path)
parser.add_argument('--library-dir', type=Path)
parser.add_argument('--single-cpu', action='store_true', help='also run contracts on one allowed Linux CPU')
parser.add_argument('--output-json', type=Path)
args = parser.parse_args()
p = Path(__file__).resolve().parents[1]
guide = p / 'Guide.md'
text = guide.read_text().split('## 2. oneTBB', 1)[1]
blocks = re.findall('```cpp\\n(.*?)\\n```', text, re.S)
temporary = tempfile.TemporaryDirectory(prefix='course-tbb-guide-')
out = Path(temporary.name)
base = [args.compiler, '-std=c++17', '-O2', '-Wall', '-Wextra', '-Werror', '-pthread']
if args.include_dir:
    base.append('-I' + str(args.include_dir.resolve()))
link_dirs = []
if args.library_dir:
    library = str(args.library_dir.resolve())
    link_dirs = ['-L' + library, '-Wl,-rpath,' + library]
records = []

def run(args, label):
    r = subprocess.run(args, capture_output=True, text=True, timeout=60)
    rec = {'label': label, 'argv': args, 'returncode': r.returncode, 'stdout': r.stdout, 'stderr': r.stderr}
    records.append(rec)
    print(label, r.returncode, r.stdout[:300], r.stderr[:1500], flush=True)
    return r

def block(marker):
    matches = [b for b in blocks if marker in b]
    if len(matches) != 1:
        raise ValueError(f'Expected one code block for {marker!r}; found {len(matches)}')
    return matches[0]
header = '#include <oneapi/tbb.h>\n#include <oneapi/tbb/flow_graph.h>\n#include <vector>\n#include <cmath>\n#include <limits>\n#include <numeric>\n#include <cstdio>\n#include <stdexcept>\n#include <algorithm>\n#include <memory>\nusing namespace oneapi::tbb;\nusing namespace oneapi::tbb::flow;\nvoid require(bool value) {if(!value) throw std::runtime_error("numerical or message mismatch");}\n'

def compile_run(name, source, libs=('-ltbb',), check=lambda r: r.returncode == 0):
    f = out / (name + '.cpp')
    f.write_text(source)
    binary = out / name
    r = run(base + [str(f), *link_dirs, *libs, '-o', str(binary)], name + ' compile')
    if r.returncode == 0:
        rr = run([str(binary)], name + ' execute')
        if not check(rr):
            return 1
    return int(r.returncode != 0)
fail = 0
fail += compile_run('squaring', block('const size_t N ='), check=lambda r: r.returncode == 0 and 'result[0]=9  result[N-1]=9' in r.stdout)
fail += compile_run('relu', block('const size_t H ='), check=lambda r: r.returncode == 0 and 'out(0,255) = 127' in r.stdout)
fail += compile_run('mse', block('double sum_sq ='), check=lambda r: r.returncode == 0 and abs(float(r.stdout.split('=')[1]) - 0.0099) < 1e-07)
fail += compile_run('allocator', block('allocated value='), ('-ltbbmalloc',), lambda r: r.returncode == 0 and r.stdout == 'allocated value=1\n')
scan = block('std::vector<float> in(N), out(N);').replace('#include "oneapi/tbb/parallel_scan.h"', '').replace('std::vector<float> in(N), out(N);', 'std::vector<float> in{1,2,3,4,5}, out(N);')
fail += compile_run('inclusive_scan', header + 'int main(){const int N=5;' + scan + '\nrequire(out==std::vector<float>({1,3,6,10,15}));}\n')
exclusive = block('write before adding — exclusive')
fail += compile_run('exclusive_scan', header + 'int main(){const int N=5;std::vector<float> in{1,2,3,4,5},out(N);' + exclusive + '\nrequire(out==std::vector<float>({0,1,3,6,10}));}\n')
compact = block('const int N = 7;')
fail += compile_run('compaction', header + 'int main(){' + compact + '\nrequire(output==std::vector<int>({1,2,3,4}));}\n')
minimum = block('struct MinResult {')
minimum = minimum.replace('struct MinResult { float val; int idx; };', '')
source = header + 'struct MinResult { float val; int idx; };\nMinResult minimum(const std::vector<float>& a){const int N=static_cast<int>(a.size());' + minimum + '\nreturn result;}\nint main(){auto a=minimum({5,-2,-2,7});require(a.val==-2&&a.idx==1);require(minimum({}).idx==-1);auto b=minimum({std::numeric_limits<float>::max(),std::numeric_limits<float>::max()});require(b.idx==0);}\n'
fail += compile_run('minimum_index', source)
reduction = block('float result = parallel_deterministic_reduce(')
fail += compile_run('deterministic_reduce', header + 'int main(){std::vector<float> a{1,2,3,4,5};const int N=5;' + reduction + '\nrequire(result==15);}\n')
flow = block('function_node<std::tuple<int,int>, continue_msg> printer').replace('#include "oneapi/tbb/flow_graph.h"', '')
fail += compile_run('graph', header + 'int main(){' + flow + '\n}\n', check=lambda r: r.returncode == 0 and r.stdout == 'square=25  cube=125\n')
indexer = block('using IndexerType =')
fail += compile_run('indexer', header + 'int got_int=0;float got_float=0;void handle_int(int v){got_int=v;}void handle_float(float v){got_float=v;}\nint main(){graph g;' + indexer + '\ng.wait_for_all();require(got_int==42&&std::abs(got_float-3.14f)<1e-6f);}\n')
pipeline = block('struct Batch {')
pipeline_support = 'std::vector<int> written; std::vector<float> load_batch_from_disk(int){return {0,255};} void run_inference(std::vector<float>& values){for(float& value:values)value*=2;} void write_predictions(int id,const std::vector<float>& values){require(values==std::vector<float>({0,2}));written.push_back(id);}\n'
fail += compile_run('ordered_pipeline', header + pipeline_support + 'int main(){' + pipeline + '\nrequire(written.size()==100);for(int i=0;i<100;++i)require(written[i]==i); }\n')
version_probe = subprocess.run(base + ['-dM', '-E', '-x', 'c++', '-include', 'oneapi/tbb/version.h', '-'], input='', text=True, capture_output=True, check=True)
header_versions = dict(re.findall('^#define (TBB_VERSION_(?:MAJOR|MINOR|PATCH)|TBB_INTERFACE_VERSION) (\\d+)$', version_probe.stdout, re.M))
binary = out / 'contracts'
r = run(base + [str(p / 'concurrency_contracts.cpp'), *link_dirs, '-ltbb', '-o', str(binary)], 'contracts compile')
fail += int(r.returncode != 0)
if r.returncode == 0:
    fail += int(run([str(binary), '--self-test'], 'contracts default affinity').returncode != 0)
    if args.single_cpu:
        if not hasattr(os, 'sched_getaffinity') or not shutil.which('taskset'):
            raise RuntimeError('--single-cpu requires Linux sched_getaffinity and taskset')
        fail += int(run(['taskset', '-c', str(min(os.sched_getaffinity(0))), str(binary), '--self-test'], 'contracts one CPU affinity').returncode != 0)
report = {'recorded_at': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'purpose': 'Independent review: actual oneTBB guide examples and scheduling contracts, not performance measurement', 'compiler': subprocess.check_output([args.compiler, '--version'], text=True).splitlines()[0], 'guide_onetbb_sections_sha256': hashlib.sha256(text.encode()).hexdigest(), 'contracts_sha256': hashlib.sha256((p / 'concurrency_contracts.cpp').read_bytes()).hexdigest(), 'onetbb_header_macros': header_versions, 'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'checks': records, 'limitations': ['Some guide fragments need a harness supplying their documented inputs and expected outputs.', 'No GPU execution. No ThreadSanitizer claim. One-CPU affinity is an additional progress check, not a proof for every scheduler.'], 'all_passed': fail == 0}
if args.output_json:
    args.output_json.write_text(json.dumps(report, indent=2) + '\n')
if fail:
    raise SystemExit('Native guide validation failed; inspect compiler output.')
print('PASS 12 guide examples and 6 concurrency contracts' + (' (including one-CPU affinity)' if args.single_cpu else ''))
temporary.cleanup()
