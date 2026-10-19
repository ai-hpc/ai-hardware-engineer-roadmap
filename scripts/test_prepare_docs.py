"""Check published source completeness in Git checkouts and extracted archives."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from prepare_docs import prepare


class PrepareDocsTests(unittest.TestCase):
    def file(self, root, name, content="source"):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        return path

    def test_archive_keeps_dependencies_and_build_assets(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            assets = ["README.md", "requirements-docs.txt", "mkdocs.yml",
                      "scripts/prepare_docs.sh", "scripts/prepare_docs.py",
                      "Assets/js/vendor/mathjax/tex-mml-chtml.js",
                      "Assets/js/vendor/mathjax/output/chtml/fonts/woff-v2/MathJax_Main-Regular.woff",
                      "Assets/js/vendor/mathjax/LICENSE",
                      "Assets/js/vendor/mathjax/README.txt",
                      "Phase 3 - AI/labs/requirements-provider.lock",
                      "Phase 5 - Systems/labs/requirements-cpu.lock",
                      "Phase 5 - Systems/labs/requirements-cpu.txt",
                      "Phase 5 - Systems/labs/model.json", "Phase 5 - Systems/labs/LICENSE",
                      "Phase 1 - CPU/labs/main.c", "Phase 1 - CPU/labs/kernel.cl",
                      "Phase 2 - Embedded/labs/Makefile", "Phase 2 - Embedded/labs/board.dts",
                      "Phase 4 - Jetson/labs/Dockerfile", "Phase 4 - Jetson/labs/telemetry.js",
                      "Projects/jetson-llm-runtime/CMakeLists.txt",
                      "Projects/jetson-llm-runtime/docs/testing.md",
                      "Phase 2 - Embedded/docs/source-guide.md",
                      "Projects/jetson-llm-runtime/scripts/test_plan.sh"]
            for asset in assets: self.file(root, asset, asset)
            font = root / "Assets/js/vendor/mathjax/output/chtml/fonts/woff-v2/MathJax_Main-Regular.woff"
            font.write_bytes(b"wOFF\x00\xff\x01")  # asset publication must preserve binary data
            expected = {asset: (root / asset).read_bytes() for asset in assets}
            self.assertEqual(prepare(root), len(assets))
            for asset in assets:
                self.assertEqual((root / 'docs' / asset).read_bytes(), expected[asset])
            self.assertEqual(prepare(root), len(assets))  # docs never copies into itself

    def test_shell_entry_runs_from_a_non_git_archive_and_another_cwd(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'archive with spaces'
            root.mkdir()
            self.file(root, 'README.md')
            (root/'scripts').mkdir()
            for name in ('prepare_docs.sh', 'prepare_docs.py'):
                shutil.copy2(Path(__file__).with_name(name), root/'scripts'/name)
            result=subprocess.run(['bash', str(root/'scripts/prepare_docs.sh')],
                                  cwd=directory, text=True, capture_output=True, timeout=10)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertTrue((root/'docs/README.md').is_file())
            self.assertIn('Prepared 3 public source files',result.stdout)

    def test_generated_and_hidden_content_is_not_published(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.file(root, 'README.md')
            for excluded in ['.codex/private.md', 'Phase 1 - CPU/.venv/test.py',
                             'Phase 1 - CPU/build/generated.py', '_site/index.md',
                             'docs/stale.md', 'node_modules/dependency.py']:
                self.file(root, excluded)
            self.assertEqual(prepare(root), 1)
            self.assertEqual(list((root/'docs').iterdir()), [root/'docs/README.md'])

    @unittest.skipUnless(shutil.which('git'), 'Git discovery requires Git')
    def test_git_preserves_ignore_rules_and_includes_new_assets(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subprocess.run(['git', 'init', '-q', str(root)], check=True)
            self.file(root, '.gitignore', 'private.py\ndocs/\n')
            self.file(root, 'README.md')
            self.file(root, 'Phase 3 - AI/labs/new.lock')
            self.file(root, 'private.py')
            self.assertEqual(prepare(root), 2)
            self.assertFalse((root/'docs/private.py').exists())

    @unittest.skipUnless(shutil.which('git'), 'Git failure requires Git')
    def test_broken_git_does_not_erase_previous_docs_or_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.file(root, '.git', 'gitdir: missing-git-directory\n')
            self.file(root, 'README.md')
            self.file(root, 'docs/previous.md', 'previous')
            with self.assertRaises(subprocess.CalledProcessError): prepare(root)
            self.assertEqual((root/'docs/previous.md').read_text(), 'previous')

    def test_empty_source_and_external_symlink_fail_before_replacing_docs(self):
        with tempfile.TemporaryDirectory() as directory, tempfile.TemporaryDirectory() as external:
            root = Path(directory)
            self.file(root, 'docs/previous.md', 'previous')
            with self.assertRaises(ValueError): prepare(root)
            target=self.file(Path(external), 'secret.py')
            (root/'escape.py').symlink_to(target)
            with self.assertRaises(ValueError): prepare(root)
            self.assertEqual((root/'docs/previous.md').read_text(), 'previous')


if __name__ == '__main__': unittest.main()
