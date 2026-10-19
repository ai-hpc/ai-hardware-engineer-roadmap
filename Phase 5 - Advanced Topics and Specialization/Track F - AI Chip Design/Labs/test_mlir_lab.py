"""Optional real compiler tests; missing tools skip, compiler failures fail."""
from pathlib import Path
import tempfile
import unittest

from run_mlir_lab import HERE, execute, run, tool


class MLIRTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.opt = tool("mlir-opt")
            tool("mlir-translate")
            tool("mlir-runner")
        except FileNotFoundError as exc:
            raise unittest.SkipTest(str(exc))

    def test_actual_lowering_translation_and_cpu_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            result = execute(HERE / "matmul.mlir", folder)
            self.assertEqual(result["result"], "415")
            self.assertIn("scf.for", (folder / "02-loops.mlir").read_text())
            self.assertNotIn("linalg.matmul", (folder / "03-llvm-dialect.mlir").read_text())
            self.assertIn("define i32 @main()", (folder / "04-matmul.ll").read_text())

    def test_wrong_operand_value_is_detected_after_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            source = folder / "wrong-value.mlir"
            source.write_text((HERE / "matmul.mlir").read_text().replace(
                "[[1.0, 2.0, 3.0]", "[[2.0, 2.0, 3.0]", 1))
            with self.assertRaisesRegex(AssertionError, "matrix validation failed"):
                execute(source, folder / "stages")

    def test_incompatible_reduction_shape_fails_verification(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "bad-shape.mlir"
            source.write_text("""module {
  func.func @bad(%a: memref<2x3xf32>, %b: memref<4x2xf32>, %c: memref<2x2xf32>) {
    linalg.matmul ins(%a, %b : memref<2x3xf32>, memref<4x2xf32>) outs(%c : memref<2x2xf32>)
    return
  }
}
""")
            with self.assertRaisesRegex(RuntimeError, "inferred input/output operand"):
                run([self.opt, str(source)])


if __name__ == "__main__":
    unittest.main()
