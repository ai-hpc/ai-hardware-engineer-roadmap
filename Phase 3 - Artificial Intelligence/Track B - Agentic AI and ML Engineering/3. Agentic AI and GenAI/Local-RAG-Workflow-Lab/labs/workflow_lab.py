"""Optional real LangGraph workflow: fan-in, approval, SQLite restart and local idempotency."""
import argparse
import hashlib
import json
import operator
from pathlib import Path
import sqlite3
from typing import Annotated, TypedDict
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command, interrupt
from local_rag import Index


class State(TypedDict):
    run_id: str
    query: str
    draft: dict
    reviews: Annotated[list[str], operator.add]
    approved: bool
    status: str


def save_once(path, run_id, payload):
    """One local DB transaction; an external service would need its own idempotency contract."""
    encoded=json.dumps(payload,sort_keys=True,separators=(',',':'))
    digest=hashlib.sha256(encoded.encode()).hexdigest()
    with sqlite3.connect(path) as db:
        db.execute('CREATE TABLE IF NOT EXISTS artifacts (run_id TEXT PRIMARY KEY, digest TEXT NOT NULL, payload TEXT NOT NULL)')
        db.execute('INSERT OR IGNORE INTO artifacts VALUES (?,?,?)',(run_id,digest,encoded))
        existing=db.execute('SELECT digest FROM artifacts WHERE run_id=?',(run_id,)).fetchone()[0]
        if existing != digest:raise ValueError('same run id with different approved content')
    return digest


def build_graph(index, artifacts, saver, crash_after_save=False):
    def retrieve(state):
        return {'draft':index.excerpt(state['query']), 'status':'drafted'}

    def citations(state):
        draft=state['draft']
        ok=bool(draft['citations']) and all(index.verify_citation(x) for x in draft['citations'])
        return {'reviews':['citations:pass' if ok else 'citations:fail']}

    def scope(state):
        ids={x['id'] for x in state['draft']['citations']}
        return {'reviews':['scope:pass' if ids and ids <= set(index.documents) else 'scope:fail']}

    def review(state):
        # No writes before interrupt: this node restarts from its beginning on resume.
        if sorted(state['reviews']) != ['citations:pass','scope:pass']:
            return {'approved':False,'status':'blocked'}
        decision=interrupt({'draft':state['draft'],'question':'Save this quotation to the local exercise artifact database?'})
        if type(decision) is not bool:raise ValueError('approval must be a real boolean')
        return {'approved':decision,'status':'approved' if decision else 'rejected'}

    def publish(state):
        save_once(artifacts,state['run_id'],state['draft'])
        if crash_after_save:raise RuntimeError('injected crash after artifact commit, before node checkpoint')
        return {'status':'saved'}

    graph=StateGraph(State)
    for name,node in [('retrieve',retrieve),('citations',citations),('scope',scope),('review',review),('publish',publish)]:
        graph.add_node(name,node)
    graph.add_edge(START,'retrieve')
    graph.add_edge('retrieve','citations');graph.add_edge('retrieve','scope')
    graph.add_edge(['citations','scope'],'review')
    graph.add_conditional_edges('review',lambda s:'publish' if s['approved'] else END)
    graph.add_edge('publish',END)
    return graph.compile(checkpointer=saver)


def run(directory, run_id, action, query='battery return threshold', crash=False):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    index=Index();config={'configurable':{'thread_id':run_id},'recursion_limit':12}
    try:
        with SqliteSaver.from_conn_string(str(directory/'checkpoints.db')) as saver:
            graph=build_graph(index,directory/'artifacts.db',saver,crash)
            saved=graph.get_state(config)
            if action=='start':
                if saved.values:raise ValueError('run id already exists; choose a new id or resume')
                value={'run_id':run_id,'query':query,'reviews':[],'approved':False,'status':'new','draft':{}}
            elif not saved.values:raise ValueError('unknown run id')
            elif action in ('approve','reject'):
                if not any(task.interrupts for task in saved.tasks):raise ValueError('no approval is pending')
                value=Command(resume=action=='approve')
            elif action=='retry':
                if any(task.interrupts for task in saved.tasks):raise ValueError('approval is pending; retry is not approval')
                value=None
            else:raise ValueError('unknown action')
            graph.invoke(value,config=config)
            snapshot=graph.get_state(config)
            return {'state':snapshot.values,'next':list(snapshot.next),
                    'pending_approval':any(task.interrupts for task in snapshot.tasks)}
    finally:index.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['start','approve','reject','retry'])
    parser.add_argument('--directory',type=Path,required=True)
    parser.add_argument('--run-id',default='lesson-1')
    parser.add_argument('--query',default='battery return threshold')
    parser.add_argument('--crash-after-save',action='store_true')
    args=parser.parse_args()
    print(json.dumps(run(args.directory,args.run_id,args.action,args.query,args.crash_after_save),indent=2))
