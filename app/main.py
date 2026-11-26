from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import runs, scenarios, templates
from app.api.catalogs import cameras_router, drones_router, persons_router
from app.application import runs_service
from app.config import settings
from app.infrastructure.db import SessionLocal
from app.infrastructure.executor import ProcessDispatcher


@asynccontextmanager
async def lifespan(app: FastAPI):
    session_factory = getattr(app.state, "session_factory", SessionLocal)
    with session_factory() as session:
        runs_service.fail_interrupted_runs(session)
    owns_dispatcher = not hasattr(app.state, "dispatcher")
    if owns_dispatcher:
        app.state.dispatcher = ProcessDispatcher(settings.max_concurrent_runs)
    yield
    if owns_dispatcher:
        with session_factory() as session:
            runs_service.request_cancel_all_active(session)
        app.state.dispatcher.shutdown()


app = FastAPI(title="RastroAéreo API", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", tags=["health"])
def health() -> dict[str, str]:
    return {"status": "ok"}


app.include_router(drones_router)
app.include_router(cameras_router)
app.include_router(persons_router)
app.include_router(scenarios.router)
app.include_router(templates.router)
app.include_router(runs.router)
