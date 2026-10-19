"""Deterministic single-worker queue model; no ROS or scheduling measurements."""
from collections import deque
import json


def queue_trace(samples=20, period_ms=20, service_ms=50, latest_only=False):
    for value,name in [(samples,'samples'),(period_ms,'period_ms'),(service_ms,'service_ms')]:
        if type(value) is not int or value <= 0:
            raise ValueError(f'{name} must be a positive integer')
    if samples > 100000:
        raise ValueError('bounded teaching fixture supports at most 100000 samples')
    if type(latest_only) is not bool:
        raise ValueError('latest_only must be Boolean')
    waiting=deque();active=None;finish=None;next_sequence=0;completed=[];dropped=[]
    while next_sequence < samples or active is not None or waiting:
        arrival=next_sequence*period_ms if next_sequence < samples else None
        now=min(value for value in (arrival,finish) if value is not None)
        # At an equal timestamp, enqueue arrivals before selecting the next job.
        if arrival == now:
            if latest_only and waiting:
                dropped.append(waiting.popleft())
            waiting.append(next_sequence);next_sequence+=1
        if finish == now:
            completed.append({'sequence':active,'capture_ms':active*period_ms,
                              'complete_ms':now,'age_ms':now-active*period_ms})
            active=finish=None
        if active is None and waiting:
            active=waiting.popleft();finish=now+service_ms
    return {'model':'one non-preemptive worker; deterministic arrivals; no middleware/network',
            'samples':samples,'period_ms':period_ms,'service_ms':service_ms,
            'latest_only':latest_only,'completed':completed,'dropped':dropped,
            'max_capture_to_complete_ms':max(item['age_ms'] for item in completed)}


if __name__=='__main__':
    print(json.dumps({'fifo':queue_trace(),'latest_pending':queue_trace(latest_only=True)},indent=2))
