from __future__ import annotations

import asyncio
import json
import time
from typing import Any

from fastapi import WebSocket
from starlette.websockets import (
    WebSocketDisconnect,
    WebSocketState,
)


class BrowserAudioOutput:
    SAMPLE_RATE = 44_100
    BYTES_PER_SAMPLE = 4

    def __init__(
        self,
        transport: "BrowserTransport",
    ) -> None:
        self._transport = transport
        self._playback_until = 0.0

    @property
    def is_playing(self) -> bool:
        return (
            time.monotonic()
            < self._playback_until
        )

    async def enqueue(
        self,
        audio: bytes,
    ) -> None:
        if not audio:
            return

        duration = (
            len(audio)
            / self.BYTES_PER_SAMPLE
            / self.SAMPLE_RATE
        )

        self._playback_until = (
            max(
                time.monotonic(),
                self._playback_until,
            )
            + duration
        )

        await self._transport.send_audio(
            audio
        )

    async def clear(self) -> None:
        self._playback_until = (
            time.monotonic()
        )

        await self._transport.send_event(
            {
                "type":
                    "clear_audio",
            }
        )

    async def wait_until_done(
        self,
    ) -> None:
        while True:
            remaining = (
                self._playback_until
                - time.monotonic()
            )

            if remaining <= 0:
                return

            await asyncio.sleep(
                min(
                    remaining,
                    0.05,
                )
            )


class BrowserTransport:
    """
    Browser transport with disconnect-safe, best-effort sends.
    """

    def __init__(
        self,
        websocket: WebSocket,
    ) -> None:
        self.websocket = websocket
        self._send_lock = (
            asyncio.Lock()
        )
        self._disconnected = False

        self.output = (
            BrowserAudioOutput(
                self
            )
        )

    @property
    def connected(self) -> bool:
        if self._disconnected:
            return False

        client_state = getattr(
            self.websocket,
            "client_state",
            None,
        )

        app_state = getattr(
            self.websocket,
            "application_state",
            None,
        )

        # Unit-test doubles do not necessarily expose Starlette state.
        if (
            client_state is None
            and app_state is None
        ):
            return True

        client_ok = (
            True
            if client_state is None
            else client_state
            is WebSocketState.CONNECTED
        )

        app_ok = (
            True
            if app_state is None
            else app_state
            is WebSocketState.CONNECTED
        )

        return (
            client_ok
            and app_ok
        )

    def mark_disconnected(
        self,
    ) -> None:
        self._disconnected = True

    async def accept(self) -> None:
        await self.websocket.accept()

    async def receive(self) -> dict:
        try:
            message = (
                await self.websocket.receive()
            )

            if (
                message.get("type")
                == "websocket.disconnect"
            ):
                self.mark_disconnected()

            return message

        except WebSocketDisconnect:
            self.mark_disconnected()

            return {
                "type":
                    "websocket.disconnect",
            }

    async def send_event(
        self,
        payload: dict[str, Any],
        **_: Any,
    ) -> bool:
        if not self.connected:
            return False

        try:
            async with self._send_lock:
                if not self.connected:
                    return False

                await self.websocket.send_text(
                    json.dumps(
                        payload,
                        ensure_ascii=False,
                        default=str,
                    )
                )

            return True

        except Exception:
            self.mark_disconnected()
            return False

    async def send_audio(
        self,
        audio: bytes,
    ) -> bool:
        if (
            not audio
            or not self.connected
        ):
            return False

        try:
            async with self._send_lock:
                if not self.connected:
                    return False

                await self.websocket.send_bytes(
                    audio
                )

            return True

        except Exception:
            self.mark_disconnected()
            return False

    async def close(
        self,
        code: int = 1000,
    ) -> None:
        if not self.connected:
            self.mark_disconnected()
            return

        try:
            await self.websocket.close(
                code=code
            )
        except Exception:
            pass

        self.mark_disconnected()
