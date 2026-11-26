from fastapi import APIRouter, HTTPException
from sqlalchemy import select

from app.api.crud import SessionDep
from app.api.schemas import TemplateRead
from app.infrastructure.models import ScenarioTemplate

router = APIRouter(prefix="/templates", tags=["templates"])


@router.get("", response_model=list[TemplateRead])
def list_templates(session: SessionDep):
    return session.scalars(select(ScenarioTemplate).order_by(ScenarioTemplate.id)).all()


@router.get("/{slug}", response_model=TemplateRead)
def get_template(slug: str, session: SessionDep):
    template = session.scalar(select(ScenarioTemplate).where(ScenarioTemplate.slug == slug))
    if template is None:
        raise HTTPException(status_code=404, detail="Plantilla no encontrada")
    return template
