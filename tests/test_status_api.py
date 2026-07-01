from fastapi.testclient import TestClient

from backend.app import app


def test_health_endpoint() -> None:
    client = TestClient(app)
    res = client.get("/api/health")
    assert res.status_code == 200
    assert res.json()["status"] == "ok"


def test_game_endpoints_round_trip() -> None:
    client = TestClient(app)

    reset = client.post("/api/game/reset")
    assert reset.status_code == 200
    state = reset.json()
    assert state["winner"] == 0

    move = client.post("/api/game/move", json={"move": 0})
    assert move.status_code == 200
    payload = move.json()
    assert "board" in payload
    assert payload["legal_moves"] < state["legal_moves"]


def test_invalid_move_rejected() -> None:
    client = TestClient(app)

    client.post("/api/game/reset")
    first = client.post("/api/game/move", json={"move": 0})
    assert first.status_code == 200

    second = client.post("/api/game/move", json={"move": 0})
    assert second.status_code == 400


def test_pause_resume_controls() -> None:
    client = TestClient(app)

    pause = client.post("/api/control/pause")
    assert pause.status_code == 200
    status1 = client.get("/api/status").json()
    assert status1["paused"] is True
    assert status1["status"] == "paused"

    resume = client.post("/api/control/resume")
    assert resume.status_code == 200
    status2 = client.get("/api/status").json()
    assert status2["paused"] is False


def test_web_game_ingestion_updates_counters() -> None:
    client = TestClient(app)

    client.post("/api/control/pause")
    before = client.get("/api/status").json()

    client.post("/api/game/reset")
    winner = 0
    for idx in range(225):
        res = client.post("/api/game/move", json={"move": idx})
        if res.status_code != 200:
            continue
        payload = res.json()
        winner = int(payload["winner"])
        if winner != 0 or int(payload["legal_moves"]) == 0:
            break

    after = client.get("/api/status").json()
    assert int(after["human_samples"]) >= int(before.get("human_samples", 0))
    assert int(after["human_games"]) >= int(before.get("human_games", 0))
    assert int(after["replay_size"]) >= int(before.get("replay_size", 0))
    client.post("/api/control/resume")
