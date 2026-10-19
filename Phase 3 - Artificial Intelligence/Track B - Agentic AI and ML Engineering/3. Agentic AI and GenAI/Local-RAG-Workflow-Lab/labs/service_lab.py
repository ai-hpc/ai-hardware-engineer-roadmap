"""Optional local teaching API. Public synthetic corpus only; no model or authentication service."""
from contextlib import asynccontextmanager
import json
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field
from local_rag import Index


class Question(BaseModel):
    model_config=ConfigDict(extra='forbid',strict=True)
    query: str=Field(min_length=1,max_length=500)
    k: int=Field(default=3,ge=1,le=5)


def create_app():
    @asynccontextmanager
    async def lifespan(app):
        app.state.index=Index()
        app.state.ready=True
        try:yield
        finally:
            app.state.ready=False
            app.state.index.close()

    app=FastAPI(title='Local source quotation exercise',lifespan=lifespan)
    app.state.ready=False

    def index():
        if not app.state.ready:raise HTTPException(503,'Index is not ready')
        return app.state.index

    @app.get('/healthz')
    def health():return {'status':'alive'}

    @app.get('/readyz')
    def ready():return {'ready':True,'snapshot':index().snapshot}

    @app.post('/ask')
    def ask(question: Question):return index().excerpt(question.query,question.k)

    @app.post('/ask/stream')
    def stream(question: Question):
        result=index().excerpt(question.query,question.k)
        def events():
            # Each payload is JSON: embedded newlines cannot inject new SSE events.
            yield 'event: answer\ndata: '+json.dumps(result)+'\n\n'
            yield 'event: done\ndata: {}\n\n'
        return StreamingResponse(events(),media_type='text/event-stream')

    return app


app=create_app()
