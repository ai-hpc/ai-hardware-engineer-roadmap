from pathlib import Path
import json
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from fastapi.testclient import TestClient
from service_lab import create_app
from langchain_text_splitters import CharacterTextSplitter,RecursiveCharacterTextSplitter


class ServiceTests(unittest.TestCase):
    def test_readiness_lifespan_and_public_answer(self):
        app=create_app()
        client=TestClient(app)
        self.assertEqual(client.get('/readyz').status_code,503)
        with client:
            self.assertEqual(client.get('/readyz').status_code,200)
            r=client.post('/ask',json={'query':'battery return threshold'})
            self.assertEqual(r.status_code,200)
            self.assertEqual(r.json()['citations'][0]['id'],'battery-v2')
        self.assertFalse(app.state.ready)

    def test_schema_bounds_identity_and_boolean_integer_rejected(self):
        with TestClient(create_app()) as client:
            for body in [{'query':''},{'query':'x'*501},{'query':'battery','k':True},{'query':'battery','k':6},{'query':'battery','roles':['operator']}]:
                self.assertEqual(client.post('/ask',json=body).status_code,422)
            self.assertTrue(client.post('/ask',json={'query':'VIOLET'}).json()['abstained'])

    def test_post_sse_framing_and_get_method_mismatch(self):
        with TestClient(create_app()) as client:
            self.assertEqual(client.get('/ask/stream').status_code,405)
            r=client.post('/ask/stream',json={'query':'camera checkerboard'})
            self.assertTrue(r.headers['content-type'].startswith('text/event-stream'))
            blocks=r.text.strip().split('\n\n')
            self.assertEqual(len(blocks),2)
            first=json.loads(blocks[0].split('data: ',1)[1])
            self.assertEqual(first['citations'][0]['id'],'camera')
            self.assertTrue(blocks[1].startswith('event: done'))

    def test_character_splitter_is_not_a_hard_limit_but_recursive_fallback_is(self):
        text='x'*31
        simple=CharacterTextSplitter(chunk_size=10,chunk_overlap=0,separator='\n')
        recursive=RecursiveCharacterTextSplitter(chunk_size=10,chunk_overlap=0,separators=['\n',' ',''])
        self.assertEqual(len(simple.split_text(text)[0]),31)
        pieces=recursive.split_text(text)
        self.assertTrue(all(len(p)<=10 for p in pieces))
        self.assertEqual(''.join(pieces),text)


if __name__=='__main__':unittest.main()
