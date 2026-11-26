import pytest

from app.application import runs_service
from app.config import settings
from app.infrastructure.executor import run_job
from app.infrastructure.models import Run
from tests.payloads import CAMERA, DRONE, PERSON, SCENARIO, THERMAL

RUN_OVERRIDES = dict(
    n_drones=2, n_points=3, population_size=12, generations=3, seed=3, random_baseline_factor=1
)


@pytest.fixture
def ids(client):
    drone = client.post("/drones", json=DRONE).json()["id"]
    camera = client.post("/cameras", json=CAMERA).json()["id"]
    thermal = client.post("/cameras", json=THERMAL).json()["id"]
    person = client.post("/persons", json=PERSON).json()["id"]
    scenario = client.post("/scenarios", json=SCENARIO).json()["id"]
    return dict(drone=drone, cameras=[camera, thermal], person=person, scenario=scenario)


def payload(ids, **overrides):
    return {
        "scenario_id": ids["scenario"],
        "person_id": ids["person"],
        "drone_ids": [ids["drone"]],
        "camera_ids": ids["cameras"],
        **RUN_OVERRIDES,
        **overrides,
    }


def test_full_run_flow(client, ids):
    created = client.post("/runs", json=payload(ids))
    assert created.status_code == 202
    run_id = created.json()["id"]

    detail = client.get(f"/runs/{run_id}").json()
    assert detail["status"] == "completed"
    assert detail["progress"] == 1.0
    assert detail["latest_generation"]["gen"] == 3
    assert detail["params"]["seed"] == 3
    assert detail["solution_count"] > 0

    generations = client.get(f"/runs/{run_id}/generations").json()
    assert [g["gen"] for g in generations] == [1, 2, 3]

    results = client.get(f"/runs/{run_id}/results").json()
    assert set(results["profiles"]) == {"best_detection", "lowest_consumption", "balanced"}
    methods = {row["method"] for row in results["comparison"]}
    assert {"ga_best_detection", "grid"} <= methods
    assert len(results["pareto"]) >= 1

    trajectories = client.get(f"/runs/{run_id}/trajectories", params={"profile": "balanced"}).json()
    assert len(trajectories["solution"]["route"]["drones"]) == 2
    assert len(trajectories["solution"]["route"]["drones"][0]["waypoints"]) == 3
    assert trajectories["scenario"]["width_m"] == 400

    solution_id = results["profiles"]["balanced"]["id"]
    solution = client.get(f"/runs/{run_id}/solutions/{solution_id}").json()
    assert len(solution["chromosome"]) == 2 * 3 * 4 + 2 * 2

    assert client.post(f"/runs/{run_id}/cancel").status_code == 409
    assert client.delete(f"/runs/{run_id}").status_code == 204
    assert client.get(f"/runs/{run_id}").status_code == 404


def test_params_default_from_settings_and_seed_is_stored(client, ids):
    body = payload(ids)
    for key in ("population_size", "generations", "seed", "n_drones", "n_points"):
        body.pop(key)
    body.update(population_size=8, generations=1, n_drones=1, n_points=2)
    created = client.post("/runs", json=body)
    params = client.get(f"/runs/{created.json()['id']}").json()["params"]
    assert params["battery_reserve"] == settings.battery_reserve
    assert params["crossover_eta"] == settings.default_crossover_eta
    assert isinstance(params["seed"], int)


def test_snapshot_survives_catalog_changes(client, ids):
    run_id = client.post("/runs", json=payload(ids)).json()["id"]
    client.put(f"/drones/{ids['drone']}", json={**DRONE, "battery_wh": 1})
    solution = client.get(f"/runs/{run_id}/trajectories").json()["solution"]
    assert solution["route"]["drones"][0]["battery_remaining_pct"] < 100
    assert solution["feasible"] in (True, False)
    assert client.get(f"/runs/{run_id}").json()["status"] == "completed"


def test_validation_errors(client, ids):
    assert client.post("/runs", json=payload(ids, scenario_id=999)).status_code == 404
    assert client.post("/runs", json=payload(ids, person_id=999)).status_code == 404
    assert client.post("/runs", json=payload(ids, drone_ids=[999])).status_code == 422
    too_big = settings.limit_max_population + 1
    assert client.post("/runs", json=payload(ids, population_size=too_big)).status_code == 422

    client.put(f"/drones/{ids['drone']}", json={**DRONE, "active": False})
    assert client.post("/runs", json=payload(ids)).status_code == 422


def test_cancel_queued_run(client, ids, monkeypatch, session_factory):
    from app.main import app

    monkeypatch.setattr(app.state.dispatcher, "submit", lambda run_id: None)
    run_id = client.post("/runs", json=payload(ids)).json()["id"]
    assert client.get(f"/runs/{run_id}").json()["status"] == "queued"

    assert client.post(f"/runs/{run_id}/cancel").json()["status"] == "cancelled"
    run_job(run_id, session_factory)
    assert client.get(f"/runs/{run_id}").json()["status"] == "cancelled"
    assert client.get(f"/runs/{run_id}/generations").json() == []


def test_cancel_running_run_stops_after_a_generation(client, ids, monkeypatch, session_factory):
    from app.main import app

    monkeypatch.setattr(app.state.dispatcher, "submit", lambda run_id: None)
    run_id = client.post("/runs", json=payload(ids, generations=50)).json()["id"]
    with session_factory() as session:
        session.get(Run, run_id).cancel_requested = True
        session.commit()
    run_job(run_id, session_factory)

    assert client.get(f"/runs/{run_id}").json()["status"] == "cancelled"
    assert len(client.get(f"/runs/{run_id}/generations").json()) == 1
    assert client.get(f"/runs/{run_id}/results").status_code == 409


def test_failed_run_records_error(client, ids, monkeypatch, session_factory):
    from app.main import app

    monkeypatch.setattr(app.state.dispatcher, "submit", lambda run_id: None)
    run_id = client.post("/runs", json=payload(ids)).json()["id"]
    with session_factory() as session:
        session.get(Run, run_id).scenario_snapshot = {}
        session.commit()
    run_job(run_id, session_factory)

    detail = client.get(f"/runs/{run_id}").json()
    assert detail["status"] == "failed"
    assert "KeyError" in detail["error"]


def test_active_runs_cannot_be_deleted_and_restart_marks_them_failed(
    client, ids, monkeypatch, session_factory
):
    from app.main import app

    monkeypatch.setattr(app.state.dispatcher, "submit", lambda run_id: None)
    run_id = client.post("/runs", json=payload(ids)).json()["id"]
    assert client.delete(f"/runs/{run_id}").status_code == 409

    with session_factory() as session:
        runs_service.fail_interrupted_runs(session)
    detail = client.get(f"/runs/{run_id}").json()
    assert detail["status"] == "failed"
    assert client.delete(f"/runs/{run_id}").status_code == 204
