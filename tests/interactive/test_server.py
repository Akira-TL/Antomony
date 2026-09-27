import asyncio
import json
from types import SimpleNamespace
from urllib.parse import urljoin
import zipfile

from fastapi import HTTPException
import pytest
from starlette.requests import Request

from mathhackson.interactive import server
from mathhackson.interactive.protocol import Control, Edit


def request(engine, origin="http://localhost:8775"):
    return Request({"type": "http", "app": SimpleNamespace(state=SimpleNamespace(engine=engine)),
                    "headers": [(b"host", b"localhost:8775"), (b"origin", origin.encode())]})


@pytest.mark.parametrize("base", ["http://localhost:8775/", "https://babelbeast.com/Antonomy/"])
def test_index_redirect_keeps_mount_path(base):
    response = asyncio.run(server.index())
    assert response.status_code == 307
    assert urljoin(base, response.headers["location"]) == base + "interactive.html"


def test_proxy_write_uses_original_host_without_trusting_forwarded_host():
    engine = object()
    scope = {"type": "http", "app": SimpleNamespace(state=SimpleNamespace(engine=engine)),
             "headers": [(b"host", b"babelbeast.com"), (b"origin", b"https://babelbeast.com")]}
    assert server.current(Request(scope), write=True) is engine
    scope["headers"] = [(b"host", b"babelbeast.com"), (b"origin", b"https://elsewhere.invalid"),
                        (b"x-forwarded-host", b"elsewhere.invalid")]
    with pytest.raises(HTTPException) as error:
        server.current(Request(scope), write=True)
    assert error.value.status_code == 403


def test_api_controls_preview_replay_and_export(live, monkeypatch, tmp_path):
    engine = server.Engine(live)
    req = request(engine)
    monkeypatch.setattr(server, "RUNS", tmp_path / "exports")
    async def exercise():
        check = await server.preview(req, Edit(kind="food", x=-8., y=-6.))
        assert check.valid and len(live.groups[0].world.foods) == 1
        assert (await server.edit_world(req, Edit(kind="food", x=-8., y=-6.))).valid
        assert (await server.control(req, Control(kind="step"))).tick == 1
        assert (await server.replay(req, 0)).tick == 0
        with pytest.raises(HTTPException) as invalid:
            await server.replay(req, 2)
        assert invalid.value.status_code == 400
        response = await server.export(req)
        path = engine.downloads[response.url.split("/")[-1]]
        with zipfile.ZipFile(path) as archive:
            frames = archive.read("frames.jsonl").splitlines()
            assert [json.loads(row)["tick"] for row in frames] == [0, 1]
            assert "adaptive/tick-0000-ant-00.npz" in archive.namelist()
        with pytest.raises(HTTPException) as forbidden:
            await server.control(request(engine, "https://elsewhere.invalid"), Control(kind="step"))
        assert forbidden.value.status_code == 403 and live.tick == 1
    asyncio.run(exercise())


def test_failed_engine_cannot_continue_partial_world(live):
    engine = server.Engine(live)
    engine.error = "test"
    with pytest.raises(HTTPException) as error:
        asyncio.run(server.control(request(engine), Control(kind="step")))
    assert error.value.status_code == 409 and live.tick == 0


def test_cached_state_is_not_changed_by_an_in_progress_world_step(live):
    engine = server.Engine(live)
    original = engine.view.model_dump_json()
    live.groups[0].world.foods[0].stock -= 1
    live.groups[0].world.ants[0].position[:] = [4., 3.]
    assert engine.view.model_dump_json() == original
    refreshed = engine.refresh().model_dump_json()
    live.groups[0].world.foods[0].stock -= 1
    assert engine.view.model_dump_json() == refreshed
