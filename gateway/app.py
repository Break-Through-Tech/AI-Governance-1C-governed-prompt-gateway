"""FastAPI layer: HTTP in, Gateway.handle, HTTP out. No pipeline logic lives here.

Run: uvicorn gateway.app:app
"""
# No `from __future__ import annotations`: FastAPI must resolve the request
# model built inside create_app at runtime, which string annotations prevent.
from contextlib import asynccontextmanager
import threading

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from scripts.governance import POLICY_VERSION

from .components import GatewayComponents, GatewayConfig, build_components
from .pipeline import Gateway
from .schemas import BlockedResponse, CompletedResponse, ErrorResponse, HealthResponse, request_model


def create_app(components: GatewayComponents | None = None) -> FastAPI:
    """Build the app. Tests inject components; otherwise real ones load at startup."""
    config = GatewayConfig.from_env()
    PromptRequest = request_model(config.max_prompt_chars)
    state = {'gateway': None if components is None else Gateway(components)}
    # FAISS and the JSONL audit log are not safe for concurrent writers, and sync
    # endpoints run in a thread pool, so requests are handled one at a time.
    lock = threading.Lock()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if state['gateway'] is None:
            state['gateway'] = Gateway(build_components(config))
        yield

    app = FastAPI(title='Governed Prompt Gateway', lifespan=lifespan)

    @app.post('/v1/prompt', responses={200: {'model': CompletedResponse},
                                       403: {'model': BlockedResponse},
                                       503: {'model': ErrorResponse}})
    def prompt(request: PromptRequest) -> JSONResponse:
        with lock:
            result = state['gateway'].handle(request.prompt, request.request_id)
        return JSONResponse(result.body.model_dump(mode='json'), status_code=result.status_code)

    @app.get('/health', responses={200: {'model': HealthResponse}, 503: {'model': HealthResponse}})
    def health() -> JSONResponse:
        # 503 only when a fail-closed stage is down, so status-code checks stop routing here.
        components = state['gateway'].c
        body = HealthResponse(status=components.status(), policy_version=POLICY_VERSION,
                              components=components.health())
        return JSONResponse(body.model_dump(mode='json'),
                            status_code=503 if body.status == 'unavailable' else 200)

    return app


app = create_app()
