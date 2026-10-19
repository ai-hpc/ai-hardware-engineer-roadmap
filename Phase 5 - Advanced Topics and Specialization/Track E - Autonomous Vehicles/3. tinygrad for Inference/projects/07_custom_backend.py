"""A real standalone C/ctypes runtime exercise, not a registered tinygrad backend."""
import argparse
import ctypes
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import tempfile

SOURCE = r'''
#include <stddef.h>
void affine_relu(const float *x, float *y, size_t n, float scale, float bias) {
  for (size_t i=0; i<n; ++i) {
    float value = x[i]*scale+bias;
    y[i] = value > 0.0f ? value : 0.0f;
  }
}
'''


def run():
    compiler=shutil.which('clang')
    if compiler is None: raise RuntimeError('Clang is required for this Linux shared-library exercise')
    if __import__('sys').platform != 'linux': raise RuntimeError('This shared-library loader was tested on Linux only')
    compiler_version=subprocess.run([compiler,'--version'],check=True,capture_output=True,text=True).stdout.splitlines()[0]
    records=[]
    with tempfile.TemporaryDirectory(prefix='course-c-runtime-') as directory:
        source=Path(directory)/'kernel.c';library=Path(directory)/'kernel.so';source.write_text(SOURCE)
        subprocess.run([compiler,'-std=c11','-O2','-fPIC','-shared',str(source),'-o',str(library)],
                       check=True,capture_output=True,text=True,timeout=30)
        loaded=ctypes.CDLL(str(library));kernel=loaded.affine_relu
        kernel.argtypes=[ctypes.POINTER(ctypes.c_float),ctypes.POINTER(ctypes.c_float),
                         ctypes.c_size_t,ctypes.c_float,ctypes.c_float]
        kernel.restype=None
        for n in [0,1,3,31,33,257]:
            for variation in [0,1]:
                values=[((i*7+variation*13)%37-18)/16 for i in range(n)]
                # Guard elements on both sides catch tail writes; result starts poisoned.
                inp=(ctypes.c_float*max(1,n))(*values)
                result=(ctypes.c_float*(n+2))(*([12345.]+[math.nan]*n+[-12345.]))
                pointer=ctypes.cast(ctypes.byref(result,ctypes.sizeof(ctypes.c_float)),ctypes.POINTER(ctypes.c_float))
                kernel(inp,pointer,n,2.,1.)
                expected=[max(2*x+1,0) for x in values]
                actual=list(result)[1:-1]
                if result[0]!=12345. or result[-1]!=-12345.:raise AssertionError('guard overwritten')
                if any(not math.isfinite(a) or abs(a-b)>1e-6 for a,b in zip(actual,expected)):
                    raise AssertionError('compiled result differs from scalar reference')
                records.append({'n':n,'variation':variation,'passed':True})
    return {'scope':'standalone C/ctypes runtime; no tinygrad registration or accelerator execution',
            'compiler':compiler_version,'source_sha256':hashlib.sha256(SOURCE.encode()).hexdigest(),
            'cases':records,'limitations':['one FP32 operation','contiguous host buffers','Linux shared object ABI']}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path);args=parser.parse_args()
    report=run();encoded=json.dumps(report,indent=2)
    if args.output:args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(encoded+'\n')
    print(encoded)


if __name__=='__main__':main()
