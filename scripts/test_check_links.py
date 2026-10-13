import subprocess
import tempfile
import unittest
from pathlib import Path

from check_links import broken_links, prose_only


class LinkCheckerTests(unittest.TestCase):
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
