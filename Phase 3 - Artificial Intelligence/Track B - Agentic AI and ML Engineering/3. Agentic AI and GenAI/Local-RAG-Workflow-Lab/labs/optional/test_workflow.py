from pathlib import Path
import json
import os
import sqlite3
import subprocess
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from workflow_lab import save_once

LAB=Path(__file__).resolve().parents[1]


class WorkflowTests(unittest.TestCase):
    def call(self,directory,action,run_id='fixture',extra=(),success=True):
        env=dict(os.environ,LANGSMITH_TRACING='false')
        proc=subprocess.run([sys.executable,str(LAB/'workflow_lab.py'),action,'--directory',str(directory),'--run-id',run_id,*extra],capture_output=True,text=True,timeout=20,env=env)
        if success:
            self.assertEqual(proc.returncode,0,proc.stderr)
            return json.loads(proc.stdout)
        self.assertNotEqual(proc.returncode,0)
        return proc

    def test_real_process_restart_and_crash_after_commit_deduplicates(self):
        with tempfile.TemporaryDirectory() as tmp:
            first=self.call(tmp,'start')
            self.assertTrue(first['pending_approval'])
            self.assertEqual(sorted(first['state']['reviews']),['citations:pass','scope:pass'])
            self.assertFalse((Path(tmp)/'artifacts.db').exists())
            failed=self.call(tmp,'approve',extra=('--crash-after-save',),success=False)
            self.assertIn('injected crash',failed.stderr)
            final=self.call(tmp,'retry')
            self.assertEqual(final['state']['status'],'saved')
            self.assertFalse(final['pending_approval'])
            with sqlite3.connect(Path(tmp)/'artifacts.db') as db:
                self.assertEqual(db.execute('SELECT COUNT(*) FROM artifacts').fetchone()[0],1)
            self.call(tmp,'start',success=False)

    def test_rejection_and_unanswerable_do_not_publish(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.call(tmp,'start')
            final=self.call(tmp,'reject');self.assertEqual(final['state']['status'],'rejected')
            bad=self.call(tmp,'start','unknown',('--query','underwater pressure rating'))
            self.assertEqual(bad['state']['status'],'blocked')
            self.assertFalse(bad['pending_approval'])
            self.assertFalse((Path(tmp)/'artifacts.db').exists())

    def test_retry_is_not_approval_and_ids_bind_content(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.call(tmp,'start')
            self.call(tmp,'retry',success=False)
            self.call(tmp,'approve','missing',success=False)
            path=Path(tmp)/'effects.db'
            save_once(path,'id',{'quote':'one'})
            save_once(path,'id',{'quote':'one'})
            with self.assertRaises(ValueError):save_once(path,'id',{'quote':'different'})


if __name__=='__main__':unittest.main()
