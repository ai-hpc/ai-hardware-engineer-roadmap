import subprocess
import shutil
import tempfile
import unittest
import sys
from pathlib import Path

from check_links import broken_links, prose_only


class LinkCheckerTests(unittest.TestCase):
    def cli(self, *args, cwd=None):
        return subprocess.run([sys.executable, str(Path(__file__).with_name('check_links.py')),
                               *map(str, args)], cwd=cwd, capture_output=True, text=True)

    def test_help_is_help_instead_of_an_empty_scan(self):
        result = self.cli('--help')
        self.assertEqual(result.returncode, 0)
        self.assertIn('usage:', result.stdout)
        self.assertIn('existing directory', result.stdout)
        self.assertNotIn('broken link(s)', result.stderr)

    def test_nonexistent_root_and_file_root_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            file = root / 'file.md'
            file.write_text('# A file is not a root directory\n')
            for invalid in (root / 'missing', file):
                with self.subTest(root=invalid):
                    result = self.cli(invalid)
                    self.assertEqual(result.returncode, 2)
                    self.assertIn('root must be an existing directory', result.stderr)
                    self.assertNotIn('0 broken link(s)', result.stderr)

    def test_unknown_cli_option_fails(self):
        result = self.cli('--hlep')
        self.assertEqual(result.returncode, 2)
        self.assertIn('unrecognized arguments', result.stderr)

    def test_explicit_and_default_valid_root_report_real_links(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'lesson.md').write_text('[missing](target.md)\n')
            explicit = self.cli(root)
            default = self.cli(cwd=root)
            for result in (explicit, default):
                self.assertEqual(result.returncode, 1)
                self.assertIn('lesson.md -> target.md', result.stdout)
                self.assertIn('1 broken link(s)', result.stderr)
            (root / 'target.md').write_text('# Present\n')
            self.assertEqual(self.cli(root).returncode, 0)

    def test_code_is_not_a_link(self):
        text = '```python\na[x](missing)\n```\ninline `b[y](missing)`\n[real](present.md)'
        self.assertEqual(prose_only(text).strip(), 'inline \n[real](present.md)')

    def test_long_fence_can_contain_short_fence(self):
        self.assertNotIn('missing', prose_only('````text\n```\na[x](missing)\n````'))

    def test_real_broken_link_and_encoded_and_angle_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'space file.md').write_text('# Here\n')
            (root / 'main.md').write_text(
                '[ok](space%20file.md)\n[ok](<space file.md>)\n'
                '[bad](missing.md)\n```python\na[x](wrong)\n```\n')
            self.assertEqual(broken_links(directory), {'main.md -> missing.md'})

    @unittest.skipUnless(shutil.which("git"), "Git-specific discovery needs Git")
    def test_untracked_markdown_is_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            subprocess.run(['git', 'init', '-q', directory], check=True)
            root = Path(directory)
            (root / '.gitignore').write_text('ignored.md\n')
            (root / 'new.md').write_text('[broken](missing.md)\n')
            (root / 'ignored.md').write_text('[broken](also-missing.md)\n')
            self.assertEqual(broken_links(directory), {'new.md -> missing.md'})


if __name__ == '__main__':
    unittest.main()
