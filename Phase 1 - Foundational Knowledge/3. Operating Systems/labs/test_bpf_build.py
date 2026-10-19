"""Build-only native/BPF evidence; never loads or attaches a BPF program."""
import ctypes.util
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

LAB = Path(__file__).resolve().parent


class BpfBuildTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.clang = shutil.which('clang-18') or shutil.which('clang')
        cls.cc = shutil.which('cc')
        cls.bpftool = shutil.which('bpftool')
        cls.readelf = shutil.which('readelf')
        cls.includes = [Path(p) for p in os.environ.get(
            'COURSE_BPF_INCLUDE', '/usr/include').split(os.pathsep)]
        missing = [name for name in ('clang', 'cc', 'bpftool', 'readelf')
                   if not getattr(cls, name)]
        if missing:
            raise unittest.SkipTest('missing build tools: ' + ', '.join(missing))
        if not any((p / 'bpf/bpf_helpers.h').is_file() for p in cls.includes):
            raise unittest.SkipTest('libbpf headers absent; set COURSE_BPF_INCLUDE if isolated')
        if not Path('/sys/kernel/btf/vmlinux').is_file():
            raise unittest.SkipTest('host kernel BTF absent; build boundary requires it')
        cls.libs = [ctypes.util.find_library(name) for name in ('bpf', 'elf', 'z')]
        if not all(cls.libs):
            raise unittest.SkipTest('libbpf/libelf/zlib runtime library missing')
        targets = subprocess.run([cls.clang, '--print-targets'], check=True,
                                 text=True, capture_output=True).stdout
        if 'bpf' not in targets:
            raise unittest.SkipTest('selected clang lacks the BPF target')
        cls.temp = tempfile.TemporaryDirectory(prefix='course-bpf-build-')
        cls.addClassCleanup(cls.temp.cleanup)
        cls.build = Path(cls.temp.name)
        with (cls.build / 'vmlinux.h').open('w') as output:
            subprocess.run([cls.bpftool, 'btf', 'dump', 'file',
                            '/sys/kernel/btf/vmlinux', 'format', 'c'],
                           check=True, stdout=output, stderr=subprocess.PIPE)
        includes = [f'-I{cls.build}', *(f'-I{p}' for p in cls.includes)]
        cls.obj = cls.build / 'ioctl_latency.bpf.o'
        subprocess.run([cls.clang, '-target', 'bpf', '-O2', '-g', '-Wall',
                        '-Wextra', '-Werror', *includes, '-c',
                        str(LAB / 'ioctl_latency.bpf.c'), '-o', str(cls.obj)],
                       check=True, capture_output=True, text=True)
        with (cls.build / 'ioctl_latency.skel.h').open('w') as output:
            subprocess.run([cls.bpftool, 'gen', 'skeleton', str(cls.obj)],
                           check=True, stdout=output, stderr=subprocess.PIPE)
        cls.exe = cls.build / 'ioctl_latency'
        subprocess.run([cls.cc, '-std=c17', '-O2', '-Wall', '-Wextra', '-Werror',
                        *includes, str(LAB / 'ioctl_latency.c'),
                        *(f'-l:{lib}' for lib in cls.libs), '-o', str(cls.exe)],
                       check=True, capture_output=True, text=True)

    def test_real_bpf_object_machine(self):
        header = subprocess.run([self.readelf, '-h', str(self.obj)], check=True,
                                text=True, capture_output=True).stdout
        self.assertIn('Linux BPF', header)

    def test_hook_and_map_sections(self):
        sections = subprocess.run([self.readelf, '-SW', str(self.obj)], check=True,
                                  text=True, capture_output=True).stdout
        for section in ('tracepoint/syscalls/sys_enter_ioctl',
                        'tracepoint/syscalls/sys_exit_ioctl',
                        'tracepoint/sched/sched_process_exit', '.maps', '.BTF'):
            self.assertIn(section, sections)

    def test_invalid_arguments_fail_before_load(self):
        for arguments in ([], ['0'], ['-1'], ['1x'], ['4294967296'],
                          ['99999999999999999999999999999999999'], ['1', '2']):
            with self.subTest(arguments=arguments):
                result = subprocess.run([str(self.exe), *arguments], text=True,
                                        capture_output=True, timeout=5)
                self.assertEqual(result.returncode, 2)
                self.assertNotIn('libbpf:', result.stderr)
                self.assertEqual(result.stdout, '')


if __name__ == '__main__':
    unittest.main()
