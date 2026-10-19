"""Verify pinned metadata hashes; never downloads or executes model weights."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request


def verify(online=False):
    evidence=json.loads(Path(__file__).with_name('model-evidence.json').read_text())
    results=[]
    for model in evidence['models']:
        if len(model['revision'])!=40:raise ValueError('immutable revision required')
        for item in model['source_files']:
            if model['revision'] not in item['url'] or item['path'] not in ('README.md','config.json','model.safetensors.index.json'):
                raise ValueError('unexpected metadata path or missing revision')
            row={'model':model['model'],'path':item['path'],'expected_sha256':item['sha256'],'mode':'online' if online else 'offline manifest only'}
            if online:
                with urllib.request.urlopen(item['url'],timeout=30) as response:
                    payload=response.read(1_000_001);row['http_status']=response.status
                if len(payload)>1_000_000:raise ValueError('metadata size limit exceeded')
                actual=hashlib.sha256(payload).hexdigest()
                if actual!=item['sha256'] or len(payload)!=item['bytes']:raise ValueError('pinned file content mismatch: '+item['url'])
                row.update(bytes=len(payload),actual_sha256=actual)
            results.append(row)
    return results

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--online',action='store_true');parser.add_argument('--output',type=Path)
    args=parser.parse_args();result=verify(args.online);text=json.dumps(result,indent=2)+'\n'
    if args.output:args.output.write_text(text)
    print(text)
