import os

os.environ["DATABASE_URL"] = "sqlite://"
os.environ.setdefault("CORS_ORIGINS", '["http://localhost:3000"]')

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.infrastructure.db import get_session
from app.infrastructure.executor import InlineDispatcher
from app.infrastructure.models import Base
from app.main import app


@pytest.fixture
def session_factory():
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    Base.metadata.create_all(engine)
    yield sessionmaker(bind=engine, expire_on_commit=False)
    engine.dispose()


@pytest.fixture
def client(session_factory):
    def override():
        with session_factory() as session:
            yield session

    app.dependency_overrides[get_session] = override
    app.state.session_factory = session_factory
    app.state.dispatcher = InlineDispatcher(session_factory)
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()
    del app.state.session_factory
    del app.state.dispatcher
