"""Actual SQLite FTS5 retrieval; deterministic source excerpts, no language model."""
from dataclasses import dataclass
from datetime import date
import argparse
import hashlib
import json
from pathlib import Path
import platform
import re
import sqlite3
import threading

HERE = Path(__file__).resolve().parent
STOP = {'a','an','the','is','are','of','to','and','what','when','how','does','for','in'}


def tokens(text):
    return [x for x in re.findall(r'[a-z0-9]+', text.lower()) if x not in STOP]


@dataclass(frozen=True)
class Principal:
    """Trusted application context, never accepted in model tool arguments."""
    subject: str
    roles: frozenset[str]


PUBLIC = Principal('classroom-reader', frozenset({'public'}))


class Index:
    def __init__(self, principal=PUBLIC, as_of='2026-10-18', corpus=HERE/'corpus.json'):
        self.principal = principal
        self.as_of = date.fromisoformat(as_of)
        raw = Path(corpus).read_bytes()
        self.snapshot = hashlib.sha256(raw).hexdigest()
        self._lock = threading.RLock()
        self.db = sqlite3.connect(':memory:', check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.execute('CREATE VIRTUAL TABLE chunks USING fts5(id UNINDEXED, title, text, tokenize="porter unicode61")')
        self.documents = {}
        # Build a physically scoped index: inaccessible text does not affect BM25 statistics.
        for row in json.loads(raw)['documents']:
            if row['access'] not in principal.roles:
                continue
            if self.as_of < date.fromisoformat(row['valid_from']):
                continue
            if row['valid_until'] and self.as_of >= date.fromisoformat(row['valid_until']):
                continue
            if row['id'] in self.documents:
                raise ValueError('duplicate document id')
            self.documents[row['id']] = row
            self.db.execute('INSERT INTO chunks VALUES (?,?,?)', (row['id'], row['title'], row['text']))
        self.db.commit()

    def close(self):
        self.db.close()

    def retrieve(self, query, k=3):
        if not isinstance(query,str) or not 1 <= len(query) <= 500:
            raise ValueError('query must contain 1..500 characters')
        if type(k) is not int or not 1 <= k <= 5:
            raise ValueError('k must be an integer in 1..5')
        terms = list(dict.fromkeys(tokens(query)))[:32]
        if not terms:
            return []
        # Input words are quoted as terms; a user cannot inject MATCH operators or SQL.
        expression = ' OR '.join('"' + term + '"' for term in terms)
        with self._lock:
            rows = self.db.execute(
                'SELECT id, bm25(chunks, 0, 2, 1) AS score FROM chunks '
                'WHERE chunks MATCH ? ORDER BY score, id LIMIT ?', (expression, k)).fetchall()
        return [{**self.documents[row['id']], 'bm25':row['score']} for row in rows]

    def excerpt(self, query, k=3):
        hits = self.retrieve(query,k)
        if not hits:
            return {'answer':'No matching authorized source was found.', 'abstained':True, 'citations':[]}
        # A verbatim quotation is a deliberately limited baseline, not generated reasoning.
        row = hits[0]
        quote = row['text']
        return {'answer':quote, 'abstained':False, 'citations':[{
            'id':row['id'], 'revision':row['revision'], 'start':0, 'end':len(quote),
            'quote':quote, 'text_sha256':hashlib.sha256(quote.encode()).hexdigest()}]}

    def verify_citation(self, citation):
        row = self.documents.get(citation.get('id'))
        if row is None or citation.get('revision') != row['revision']:
            return False
        start,end = citation.get('start'),citation.get('end')
        if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(row['text']):
            return False
        quote = row['text'][start:end]
        return quote == citation.get('quote') and hashlib.sha256(row['text'].encode()).hexdigest() == citation.get('text_sha256')

    def cache_key(self, query, k=3):
        context = [self.principal.subject, sorted(self.principal.roles), str(self.as_of), self.snapshot, 'fts5-porter-v1', 'verbatim-v1', k, query]
        return hashlib.sha256(json.dumps(context,separators=(',',':')).encode()).hexdigest()


def dispatch(index, call):
    """Allow exactly one read-only tool and validate arguments at execution time."""
    if not isinstance(call,dict) or set(call) != {'name','arguments'} or call['name'] != 'search_manual':
        raise ValueError('unknown tool or malformed call')
    args = call['arguments']
    if not isinstance(args,dict) or set(args) != {'query','k'}:
        raise ValueError('tool requires exactly query and k; identity is not a tool argument')
    return index.retrieve(args['query'],args['k'])


def evaluate(index, cases, k):
    rows=[]
    for case in cases:
        hits=index.retrieve(case['question'],k);ids=[x['id'] for x in hits]
        relevant=set(case['relevant'])
        recall=len(relevant.intersection(ids))/len(relevant) if relevant else None
        rr=next((1/(i+1) for i,x in enumerate(ids) if x in relevant),0.) if relevant else None
        rows.append({**case,'retrieved':ids,'recall':recall,'reciprocal_rank':rr,
                     'correct_abstention':not ids if not relevant else None})
    answerable=[r for r in rows if r['recall'] is not None]
    unanswerable=[r for r in rows if r['correct_abstention'] is not None]
    return {'k':k,'cases':rows,'answerable_count':len(answerable),'unanswerable_count':len(unanswerable),
            'macro_recall':sum(r['recall'] for r in answerable)/len(answerable) if answerable else None,
            'mrr':sum(r['reciprocal_rank'] for r in answerable)/len(answerable) if answerable else None,
            'abstention_accuracy':sum(r['correct_abstention'] for r in unanswerable)/len(unanswerable) if unanswerable else None}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--query');p.add_argument('--k',type=int,default=3);p.add_argument('--split',choices=['development','test'],default='development')
    args=p.parse_args();index=Index()
    try:
        if args.query: result=index.excerpt(args.query,args.k)
        else:
            cases=json.loads((HERE/'queries.json').read_text())[args.split]
            result={'python':platform.python_version(),'sqlite':sqlite3.sqlite_version,'corpus_sha256':index.snapshot,
                    'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    'split':args.split,'result':evaluate(index,cases,args.k),
                    'limitations':['synthetic fixture; not a generalization benchmark','no model or embedding inference','a valid quote does not prove relevance or complete support']}
        print(json.dumps(result,indent=2))
    finally:index.close()


if __name__=='__main__':main()
