"""Tests for POST /niches (see app/main.py add_niche + app/niches.py
add_custom_niche) — LLM-generated niche creation, persisted to
custom_niches.json and merged live into the NICHES registry.
Run with: pytest tests/test_add_niche.py
"""
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.config import settings
from app.main import app
from app.niches import NICHES, CUSTOM_NICHES_PATH, _load_custom_niches_file

client = TestClient(app)

_GENERATED = {
    "key": "tennis",
    "display_name": "Tennis",
    "system_context": "This content is strictly about TENNIS...",
    "banned_terms": ["basketball", "soccer", "golf"],
}


def _cleanup(key: str):
    NICHES.pop(key, None)
    data = _load_custom_niches_file()
    if key in data:
        del data[key]
        CUSTOM_NICHES_PATH.write_text(__import__("json").dumps(data), encoding="utf-8")


def test_add_niche_generates_and_persists(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "clips_dir", tmp_path)
    _cleanup("tennis")
    try:
        with patch("app.main.generate_json", return_value=_GENERATED) as mock_generate:
            response = client.post("/niches", json={"name": "Tennis"})
        assert response.status_code == 200
        assert response.json() == {"key": "tennis", "display_name": "Tennis", "parent_key": None}
        mock_generate.assert_called_once()

        # Merged into the live registry with no restart.
        assert "tennis" in NICHES
        assert NICHES["tennis"].banned_terms == ["basketball", "soccer", "golf"]

        # Persisted to disk.
        on_disk = _load_custom_niches_file()
        assert on_disk["tennis"]["display_name"] == "Tennis"

        # A ready-made local library folder is created and linked automatically.
        assert NICHES["tennis"].local_library_path == "tennis"
        assert (tmp_path / "local" / "tennis").is_dir()
    finally:
        _cleanup("tennis")


def test_add_niche_rejects_duplicate():
    with patch("app.main.generate_json") as mock_generate:
        response = client.post("/niches", json={"name": "Trucks"})
    assert response.status_code == 409
    mock_generate.assert_not_called()


def test_add_niche_rejects_blank_name():
    response = client.post("/niches", json={"name": "   "})
    assert response.status_code == 400


_SUB_GENERATED = {
    "key": "players",
    "display_name": "Tennis Players",
    "system_context": "This content is strictly about TENNIS PLAYERS...",
    "banned_terms": ["golf", "soccer"],  # "soccer" overlaps the parent's list
}


def test_add_niche_rejects_unknown_parent():
    with patch("app.main.generate_json") as mock_generate:
        response = client.post("/niches", json={"name": "Players", "parent_key": "not_a_real_niche"})
    assert response.status_code == 400
    mock_generate.assert_not_called()


def test_add_niche_creates_sub_niche_and_merges_banned_terms(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "clips_dir", tmp_path)
    _cleanup("tennis")
    _cleanup("players")
    try:
        with patch("app.main.generate_json", return_value=_GENERATED):
            client.post("/niches", json={"name": "Tennis"})

        with patch("app.main.generate_json", return_value=_SUB_GENERATED) as mock_generate:
            response = client.post("/niches", json={"name": "Players", "parent_key": "tennis"})
        assert response.status_code == 200
        body = response.json()
        assert body == {"key": "players", "display_name": "Tennis Players", "parent_key": "tennis"}

        prompt_used = mock_generate.call_args[0][0]
        assert "SUB-CATEGORY" in prompt_used
        assert "Tennis" in prompt_used

        # Union of parent's ["basketball", "soccer", "golf"] and this
        # niche's ["golf", "soccer"], deduplicated — never less restrictive
        # than the parent.
        assert sorted(NICHES["players"].banned_terms) == ["basketball", "golf", "soccer"]
        assert NICHES["players"].parent_key == "tennis"

        niches_list = client.get("/niches").json()
        keys_in_order = [n["key"] for n in niches_list]
        assert keys_in_order.index("players") == keys_in_order.index("tennis") + 1
    finally:
        _cleanup("players")
        _cleanup("tennis")


def test_delete_niche_removes_custom_niche(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "clips_dir", tmp_path)
    _cleanup("zzz_del_ep")
    try:
        with patch("app.main.generate_json", return_value={**_GENERATED, "key": "zzz_del_ep"}):
            client.post("/niches", json={"name": "Zzz Del Ep"})
        response = client.delete("/niches/zzz_del_ep")
        assert response.status_code == 200
        assert "zzz_del_ep" not in NICHES
    finally:
        _cleanup("zzz_del_ep")


def test_delete_niche_rejects_builtin():
    response = client.delete("/niches/trucks")
    assert response.status_code == 400
    assert "trucks" in NICHES


def test_rename_niche_updates_display_name(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "clips_dir", tmp_path)
    _cleanup("zzz_rename_ep")
    try:
        with patch("app.main.generate_json", return_value={**_GENERATED, "key": "zzz_rename_ep"}):
            client.post("/niches", json={"name": "Zzz Rename Ep"})
        response = client.patch("/niches/zzz_rename_ep", json={"display_name": "Renamed"})
        assert response.status_code == 200
        assert response.json()["display_name"] == "Renamed"
        assert NICHES["zzz_rename_ep"].display_name == "Renamed"
    finally:
        _cleanup("zzz_rename_ep")


def test_rename_niche_rejects_builtin():
    response = client.patch("/niches/trucks", json={"display_name": "Nope"})
    assert response.status_code == 400


def test_sync_library_rejects_niche_with_no_local_library():
    # A custom niche always gets a local library folder auto-created now
    # (see app.niches.add_custom_niche), so this exercises the 400 path via
    # a built-in niche that genuinely has none configured instead.
    assert NICHES["trucks"].local_library_path == ""
    response = client.post("/niches/trucks/sync-library")
    assert response.status_code == 400


def test_sync_library_rejects_unknown_niche():
    response = client.post("/niches/not_a_real_niche/sync-library")
    assert response.status_code == 404


def test_sync_library_runs_as_background_job_and_reports_progress(tmp_path, monkeypatch):
    """Sync must not block the request (see the Stop button's requirement to
    interrupt a run in progress) — it starts a background job and hands
    back a sync_key pollable via the existing GET /progress/{key}."""
    monkeypatch.setattr(settings, "clips_dir", tmp_path)
    _cleanup("zzz_sync_ep")
    try:
        with patch("app.main.generate_json", return_value={**_GENERATED, "key": "zzz_sync_ep"}):
            client.post("/niches", json={"name": "Zzz Sync Ep"})

        def _fake_index_niche(local_library_path, stats):
            stats.update(indexed=3, total_new=3, stopped=False)
            return {"a.mp4": {}, "b.mp4": {}, "c.mp4": {}}

        with patch("app.main.index_niche", side_effect=_fake_index_niche) as mock_index:
            response = client.post("/niches/zzz_sync_ep/sync-library")
        assert response.status_code == 200
        sync_key = response.json()["sync_key"]
        mock_index.assert_called_once_with("zzz_sync_ep", {"indexed": 3, "total_new": 3, "stopped": False})

        progress_response = client.get(f"/progress/{sync_key}")
        assert progress_response.status_code == 200
        data = progress_response.json()
        assert data["done"] is True
        assert data["result"] == {"indexed_new": 3, "total_new": 3, "stopped": False}
    finally:
        _cleanup("zzz_sync_ep")


def test_sync_library_stop_route_requests_stop(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "clips_dir", tmp_path)
    _cleanup("zzz_stop_ep")
    try:
        with patch("app.main.generate_json", return_value={**_GENERATED, "key": "zzz_stop_ep"}):
            client.post("/niches", json={"name": "Zzz Stop Ep"})
        with patch("app.main.request_stop") as mock_request_stop:
            response = client.post("/niches/zzz_stop_ep/sync-library/stop")
        assert response.status_code == 200
        assert response.json() == {"stopping": True}
        mock_request_stop.assert_called_once_with("zzz_stop_ep")
    finally:
        _cleanup("zzz_stop_ep")


def test_sync_library_stop_route_rejects_niche_with_no_local_library():
    response = client.post("/niches/trucks/sync-library/stop")
    assert response.status_code == 400


def test_sync_library_stop_route_rejects_unknown_niche():
    response = client.post("/niches/not_a_real_niche/sync-library/stop")
    assert response.status_code == 404


if __name__ == "__main__":
    # Tests that create a niche now need the tmp_path/monkeypatch pytest
    # fixtures (see app.niches.add_custom_niche's folder auto-creation) and
    # so only run under pytest — this manual runner covers the rest.
    test_add_niche_rejects_duplicate()
    test_add_niche_rejects_blank_name()
    test_add_niche_rejects_unknown_parent()
    test_delete_niche_rejects_builtin()
    test_rename_niche_rejects_builtin()
    test_sync_library_rejects_niche_with_no_local_library()
    test_sync_library_rejects_unknown_niche()
    test_sync_library_stop_route_rejects_niche_with_no_local_library()
    test_sync_library_stop_route_rejects_unknown_niche()
    print("OK")
