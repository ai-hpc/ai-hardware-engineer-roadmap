"""Compile and execute actual C; mutations must be rejected by the same checks."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

HERE = Path(__file__).resolve().parent

@unittest.skipUnless(shutil.which("cc"), "native C exercise requires a C compiler")
class RequestProtocolTests(unittest.TestCase):
    def compile_and_run(self, source=None, sanitize=False):
        with tempfile.TemporaryDirectory(prefix="course-spe-protocol-") as temporary:
            work=Path(temporary)
            implementation=work/"request_protocol.c"
            implementation.write_text(source if source is not None else (HERE/"request_protocol.c").read_text())
            binary=work/"verify"
            flags=["-std=c11","-Wall","-Wextra","-Werror","-pedantic","-O1" if sanitize else "-O2"]
            if sanitize:
                flags += ["-fsanitize=address,undefined","-fno-omit-frame-pointer","-fno-pie","-no-pie"]
            subprocess.run(["cc",*flags,"-I",str(HERE),str(implementation),str(HERE/"verify_protocol.c"),"-o",str(binary)],check=True,capture_output=True,text=True,timeout=30)
            env=os.environ.copy()
            env["ASAN_OPTIONS"]="detect_leaks=1:halt_on_error=1"
            env["UBSAN_OPTIONS"]="halt_on_error=1"
            return subprocess.run([str(binary)],capture_output=True,text=True,timeout=10,env=env)

    def test_native_protocol_contracts(self):
        result=self.compile_and_run()
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        self.assertIn("100 deadline boundary cases",result.stdout)

    def test_native_sanitized_protocol_contracts(self):
        result=self.compile_and_run(sanitize=True)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)

    def test_wraparound_mutant_is_detected(self):
        original=(HERE/"request_protocol.c").read_text()
        mutant=original.replace("(uint32_t)(now - c->started) < c->timeout", "now < (uint32_t)(c->started + c->timeout)")
        self.assertNotEqual(original,mutant)
        result=self.compile_and_run(mutant)
        self.assertNotEqual(result.returncode,0)
        self.assertIn("check failed",result.stderr)

    def test_epoch_mutant_is_detected(self):
        original=(HERE/"request_protocol.c").read_text()
        mutant=original.replace("m.kind != COURSE_REPLY || m.epoch != c->epoch ||", "m.kind != COURSE_REPLY ||")
        self.assertNotEqual(original,mutant)
        result=self.compile_and_run(mutant)
        self.assertNotEqual(result.returncode,0)
        self.assertIn("check failed",result.stderr)

if __name__=="__main__":
    unittest.main()
