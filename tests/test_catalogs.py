import pytest

from tests.payloads import CAMERA, DRONE, PERSON

CASES = {
    "drones": (DRONE, {**DRONE, "v_min": 12, "v_max": 3}, ("battery_wh", 120)),
    "cameras": (CAMERA, {**CAMERA, "modality": "infrared"}, ("focal_length_mm", 4.0)),
    "persons": (PERSON, {**PERSON, "critical_dim_m": 0}, ("px_min_optical", 25)),
}


@pytest.mark.parametrize("resource", CASES)
def test_crud_lifecycle(client, resource):
    valid, _, (field, value) = CASES[resource]
    created = client.post(f"/{resource}", json=valid)
    assert created.status_code == 201
    item_id = created.json()["id"]

    assert len(client.get(f"/{resource}").json()) == 1
    assert client.get(f"/{resource}/{item_id}").json()["name"] == valid["name"]

    updated = client.put(f"/{resource}/{item_id}", json={**valid, field: value})
    assert updated.status_code == 200
    assert updated.json()[field] == value

    assert client.delete(f"/{resource}/{item_id}").status_code == 204
    assert client.get(f"/{resource}/{item_id}").status_code == 404


@pytest.mark.parametrize("resource", CASES)
def test_duplicate_name_returns_conflict(client, resource):
    valid = CASES[resource][0]
    assert client.post(f"/{resource}", json=valid).status_code == 201
    assert client.post(f"/{resource}", json=valid).status_code == 409


@pytest.mark.parametrize("resource", CASES)
def test_invalid_payload_returns_422(client, resource):
    assert client.post(f"/{resource}", json=CASES[resource][1]).status_code == 422


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}
