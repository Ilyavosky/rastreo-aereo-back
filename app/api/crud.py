from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.infrastructure.db import get_session

SessionDep = Annotated[Session, Depends(get_session)]


def build_crud_router(
    *, prefix: str, tag: str, model: type[Any], create_schema: type[Any], read_schema: type[Any]
) -> APIRouter:
    router = APIRouter(prefix=prefix, tags=[tag])

    def fetch(session: Session, item_id: int):
        item = session.get(model, item_id)
        if item is None:
            raise HTTPException(status_code=404, detail="Recurso no encontrado")
        return item

    def commit(session: Session) -> None:
        try:
            session.commit()
        except IntegrityError:
            session.rollback()
            raise HTTPException(
                status_code=409, detail="Ya existe un registro con ese nombre"
            ) from None

    @router.get("", response_model=list[read_schema])
    def list_items(session: SessionDep):
        return session.scalars(select(model).order_by(model.id)).all()

    @router.get("/{item_id}", response_model=read_schema)
    def get_item(item_id: int, session: SessionDep):
        return fetch(session, item_id)

    @router.post("", response_model=read_schema, status_code=201)
    def create_item(payload: create_schema, session: SessionDep):
        item = model(**payload.model_dump())
        session.add(item)
        commit(session)
        session.refresh(item)
        return item

    @router.put("/{item_id}", response_model=read_schema)
    def update_item(item_id: int, payload: create_schema, session: SessionDep):
        item = fetch(session, item_id)
        for key, value in payload.model_dump().items():
            setattr(item, key, value)
        commit(session)
        session.refresh(item)
        return item

    @router.delete("/{item_id}", status_code=204)
    def delete_item(item_id: int, session: SessionDep):
        session.delete(fetch(session, item_id))
        session.commit()

    return router
