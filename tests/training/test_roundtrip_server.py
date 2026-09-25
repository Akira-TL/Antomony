import asyncio

import pytest
from fastapi import HTTPException, Request

from mathhackson.training import roundtrip_server
from mathhackson.training.schemas import RoundTripCommand


def test_roundtrip_http_control_starts_paused_and_steps_once(tmp_path, monkeypatch):
    monkeypatch.setattr(roundtrip_server, "ROOT", tmp_path)
    request = Request({"type": "http", "headers": [], "server": ("127.0.0.1", 8772)})

    async def check() -> None:
        async with roundtrip_server.lifespan(roundtrip_server.app):
            before = await roundtrip_server.state()
            assert before.paused and before.tick == 0
            assert before.phase == "memory"
            after = await roundtrip_server.command(RoundTripCommand(action="step"), request)
            assert after.paused and after.tick == 1
            assert after.release_home_probability > after.release_food_probability
            with pytest.raises(HTTPException) as error:
                await roundtrip_server.command(RoundTripCommand(action="write_mode",
                                                                 write_mode="always"), request)
            assert error.value.status_code == 409

    asyncio.run(check())
