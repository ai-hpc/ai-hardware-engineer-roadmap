"""Execute the actual shared JavaScript state model in Node; browser checks are separate."""
from pathlib import Path
import shutil
import subprocess
import unittest

MODEL = Path(__file__).with_name('telemetry.js')


@unittest.skipUnless(shutil.which('node'), 'Node required for actual JavaScript model execution')
class TelemetryModel(unittest.TestCase):
    def run_case(self, body):
        prelude = "const assert=require('node:assert/strict');const m=require(process.argv[1]);let s=m.create('a');const msg=(seq=0,time=100,distance=1)=>({session:'a',sequence:seq,captured_ms:time,distance_m:distance});\n"
        result = subprocess.run(['node', '-e', prelude+body, str(MODEL)], capture_output=True, text=True, timeout=5)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_capture_age_includes_known_delivery_delay(self):
        self.run_case("s=m.receive(s,msg(),300);assert.equal(m.view(s,350).status,'Fresh observation');assert.equal(m.view(s,350.01).status,'Stale observation');")

    def test_pause_does_not_refresh_old_sample(self):
        self.run_case("s=m.receive(s,msg(),100);const a=m.view(s,200),b=m.view(s,400);assert.equal(a.age_ms,100);assert.equal(b.age_ms,300);assert.equal(b.current,false);assert.equal(s.sample.sequence,0);")

    def test_invalid_numbers_and_future_time_rejected(self):
        self.run_case("for(const x of [NaN,Infinity,-1,21,'1',true]){const t=m.receive(s,msg(0,100,x),100);assert.equal(t.sample,null);assert.equal(m.view(t,100).status,'Invalid sample');}assert.equal(m.receive(s,msg(0,101),100).sample,null);")

    def test_duplicates_old_session_and_time_reversal_rejected(self):
        self.run_case("s=m.receive(s,msg(2,100),100);for(const q of [msg(2,100),msg(1,100),msg(3,99),{...msg(3,101),session:'old'}]){const t=m.receive(s,q,110);assert.equal(t.sample.sequence,2);assert.equal(m.view(t,110).current,false);}assert.equal(m.view(s,99).status,'Clock error');")

    def test_disconnect_and_new_session_clear_old_data(self):
        self.run_case("s=m.receive(s,msg(),100);s=m.disconnect(s);assert.equal(m.view(s,100).status,'Disconnected');assert.equal(m.receive(s,msg(1,101),101),s);s=m.create('b');assert.equal(m.view(s,102).status,'Waiting for data');assert.equal(m.receive(s,msg(),102).sample,null);")

    def test_invalid_is_recoverable_only_by_new_valid_sample(self):
        self.run_case("s=m.receive(s,msg(),100);s=m.receive(s,msg(1,101,NaN),101);assert.equal(m.view(s,102).status,'Invalid sample');s=m.receive(s,msg(2,103),103);assert.equal(m.view(s,104).status,'Fresh observation');assert.equal(s.rejected,1);")

    def test_received_sample_cannot_be_mutated_by_source(self):
        self.run_case("const q=msg();s=m.receive(s,q,100);q.distance_m=9;assert.equal(s.sample.distance_m,1);assert.ok(Object.isFrozen(s.sample));assert.throws(()=>m.view(s,100,0));assert.throws(()=>m.create(''));")


if __name__ == '__main__': unittest.main()
