from datetime import datetime

from sqlalchemy import (
    JSON,
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.infrastructure.db import Base

JsonType = JSON().with_variant(JSONB(), "postgresql")


class Drone(Base):
    __tablename__ = "drones"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(80), unique=True)
    battery_wh: Mapped[float]
    v_min: Mapped[float]
    v_max: Mapped[float]
    h_min: Mapped[float]
    h_max: Mapped[float]
    v_vertical: Mapped[float]
    p_base_w: Mapped[float]
    c_drag: Mapped[float]
    c_altitude_j_m: Mapped[float]
    active: Mapped[bool] = mapped_column(Boolean, default=True)


class Camera(Base):
    __tablename__ = "cameras"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(80), unique=True)
    modality: Mapped[str] = mapped_column(String(16))
    fov_deg: Mapped[float]
    n_px_width: Mapped[int] = mapped_column(Integer)
    power_w: Mapped[float]
    battery_wh: Mapped[float]
    max_range_m: Mapped[float]
    active: Mapped[bool] = mapped_column(Boolean, default=True)


class Person(Base):
    __tablename__ = "persons"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(80), unique=True)
    posture: Mapped[str] = mapped_column(String(16))
    critical_dim_m: Mapped[float]
    px_min_optical: Mapped[int] = mapped_column(Integer)
    px_min_thermal: Mapped[int] = mapped_column(Integer)


class Scenario(Base):
    __tablename__ = "scenarios"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(120), unique=True)
    config: Mapped[dict] = mapped_column(JsonType)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class ScenarioTemplate(Base):
    __tablename__ = "scenario_templates"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    slug: Mapped[str] = mapped_column(String(80), unique=True)
    name: Mapped[str] = mapped_column(String(120))
    description: Mapped[str] = mapped_column(Text)
    config: Mapped[dict] = mapped_column(JsonType)


class Run(Base):
    __tablename__ = "runs"
    __table_args__ = (Index("ix_runs_status", "status"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    scenario_id: Mapped[int | None] = mapped_column(
        ForeignKey("scenarios.id", ondelete="SET NULL"), nullable=True
    )
    status: Mapped[str] = mapped_column(String(16))
    cancel_requested: Mapped[bool] = mapped_column(Boolean, default=False)
    progress: Mapped[float] = mapped_column(Float, default=0.0)
    params: Mapped[dict] = mapped_column(JsonType)
    scenario_snapshot: Mapped[dict] = mapped_column(JsonType)
    catalog_snapshot: Mapped[dict] = mapped_column(JsonType)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class Generation(Base):
    __tablename__ = "generations"

    run_id: Mapped[int] = mapped_column(ForeignKey("runs.id", ondelete="CASCADE"), primary_key=True)
    gen: Mapped[int] = mapped_column(Integer, primary_key=True)
    n_feasible: Mapped[int] = mapped_column(Integer)
    best_r_de: Mapped[float | None] = mapped_column(Float, nullable=True)
    best_t_mission_min: Mapped[float | None] = mapped_column(Float, nullable=True)
    best_e_total_wh: Mapped[float | None] = mapped_column(Float, nullable=True)
    hypervolume: Mapped[float | None] = mapped_column(Float, nullable=True)


class Solution(Base):
    __tablename__ = "solutions"
    __table_args__ = (
        CheckConstraint("r_de >= 0", name="ck_solutions_r_de"),
        Index("ix_solutions_run_profile", "run_id", "profile"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("runs.id", ondelete="CASCADE"))
    profile: Mapped[str] = mapped_column(String(32))
    feasible: Mapped[bool] = mapped_column(Boolean)
    constraint_violation: Mapped[float] = mapped_column(Float)
    r_de: Mapped[float] = mapped_column(Float)
    t_mission_min: Mapped[float] = mapped_column(Float)
    e_total_wh: Mapped[float] = mapped_column(Float)
    area_explored_pct: Mapped[float] = mapped_column(Float)
    area_detected_pct: Mapped[float] = mapped_column(Float)
    chromosome: Mapped[list] = mapped_column(JsonType)
    route: Mapped[dict] = mapped_column(JsonType)
