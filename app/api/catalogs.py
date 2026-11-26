from app.api.crud import build_crud_router
from app.api.schemas import (
    CameraCreate,
    CameraRead,
    DroneCreate,
    DroneRead,
    PersonCreate,
    PersonRead,
)
from app.infrastructure.models import Camera, Drone, Person

drones_router = build_crud_router(
    prefix="/drones",
    tag="drones",
    model=Drone,
    create_schema=DroneCreate,
    read_schema=DroneRead,
)
cameras_router = build_crud_router(
    prefix="/cameras",
    tag="cameras",
    model=Camera,
    create_schema=CameraCreate,
    read_schema=CameraRead,
)
persons_router = build_crud_router(
    prefix="/persons",
    tag="persons",
    model=Person,
    create_schema=PersonCreate,
    read_schema=PersonRead,
)
