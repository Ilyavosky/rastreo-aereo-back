import numpy as np

from app.infrastructure.models import ScenarioTemplate
from tests.payloads import SCENARIO, SCENARIO_CONFIG


def test_scenario_crud_and_priority_map(client):
    created = client.post("/scenarios", json=SCENARIO)
    assert created.status_code == 201
    scenario_id = created.json()["id"]
    assert client.get(f"/scenarios/{scenario_id}").json()["config"]["width_m"] == 400

    grid = client.get(f"/scenarios/{scenario_id}/priority-map").json()
    assert grid["rows"] == 20 and grid["cols"] == 20
    assert np.isclose(np.array(grid["values"]).sum(), 1.0, atol=1e-4)

    assert client.delete(f"/scenarios/{scenario_id}").status_code == 204


def test_scenario_rejects_base_inside_forbidden_zone(client):
    config = {**SCENARIO_CONFIG, "base": {"x": 350, "y": 350}}
    response = client.post("/scenarios", json={"name": "Mala", "config": config})
    assert response.status_code == 422


def test_scenario_rejects_point_outside_map(client):
    config = {**SCENARIO_CONFIG, "last_seen": {"x": 900, "y": 10}}
    assert client.post("/scenarios", json={"name": "Mala", "config": config}).status_code == 422


def test_scenario_rejects_too_many_cells(client):
    config = {**SCENARIO_CONFIG, "width_m": 100000, "height_m": 100000, "cell_size_m": 1}
    assert client.post("/scenarios", json={"name": "Mala", "config": config}).status_code == 422


def test_templates_are_read_only_catalog(client, session_factory):
    with session_factory() as session:
        session.add(
            ScenarioTemplate(
                slug="demo", name="Demo", description="Plantilla de prueba", config=SCENARIO_CONFIG
            )
        )
        session.commit()
    assert len(client.get("/templates").json()) == 1
    assert client.get("/templates/demo").json()["name"] == "Demo"
    assert client.get("/templates/otra").status_code == 404
