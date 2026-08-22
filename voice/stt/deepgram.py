from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Awaitable, Callable
from urllib.parse import urlencode

from websockets.asyncio.client import connect
from websockets.exceptions import ConnectionClosed


from voice.language_detect import majority_word_language
logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class TranscriptEvent:
    text: str
    is_final: bool
    speech_final: bool
    confidence: float | None = None
    # Majority language of Deepgram per-word tags ("english"/"hindi") or
    # None when the provider did not tag words. Used only as a hint by
    # voice.language_detect; never controls the answer language alone.
    detected_language: str | None = None


TranscriptCallback = Callable[
    [TranscriptEvent],
    Awaitable[None],
]


class DeepgramStreamingSTT:
    """
    INPUT language is intentionally independent from UI answer language.

    The user may speak English, Hindi, or mixed Hindi/English.
    Therefore Deepgram always runs multilingual recognition.

    UI answer language is enforced later by ResponseLocalizer.
    """

    VALID_MODES = {
        "auto",
        "english",
        "hindi",
        "hinglish",
    }

    def __init__(
        self,
        *,
        api_key: str,
        transcript_callback: TranscriptCallback,
        model: str = "nova-3",
        language_mode: str = "auto",
        sample_rate: int = 16_000,
    ) -> None:
        self._api_key = api_key.strip()
        self._callback = transcript_callback
        self._model = model
        self._sample_rate = sample_rate

        self._ws = None
        self._receive_task: asyncio.Task[None] | None = None
        self._keepalive_task: asyncio.Task[None] | None = None
        self._send_lock = asyncio.Lock()
        self._running = False

        self._requested_mode = "auto"

        self.set_language_mode(
            language_mode
        )

    @property
    def running(self) -> bool:
        return self._running

    @property
    def language_mode(self) -> str:
        # Actual STT mode.
        return "auto"

    def set_language_mode(
        self,
        mode: str,
    ) -> str:
        mode = str(
            mode
            or "auto"
        ).strip().lower()

        if mode not in self.VALID_MODES:
            raise ValueError(
                "Language must be auto, english, hindi, or hinglish."
            )

        if self._running:
            raise RuntimeError(
                "Cannot change STT configuration while Deepgram is running."
            )

        self._requested_mode = mode

        logger.info(
            "Deepgram input recognition | requested_output_mode=%s | stt_language=multi",
            mode,
        )

        return mode

    def _url(self) -> str:
        query = urlencode(
            {
                "model": self._model,
                "language": "multi",
                "encoding": "linear16",
                "sample_rate": str(
                    self._sample_rate
                ),
                "channels": "1",
                "interim_results": "true",
                "smart_format": "true",
                "punctuate": "true",
                "endpointing": "300",
            }
        )

        return (
            "wss://api.deepgram.com/v1/listen?"
            + query
        )

    async def start(self) -> None:
        if self._running:
            return

        if not self._api_key:
            raise RuntimeError(
                "DEEPGRAM_API_KEY is empty."
            )

        logger.info(
            "Connecting directly to Deepgram | input_language=multi"
        )

        self._ws = await connect(
            self._url(),
            additional_headers={
                "Authorization":
                    f"Token {self._api_key}"
            },
            compression=None,
            max_size=None,
            ping_interval=20,
            ping_timeout=20,
        )

        self._running = True

        self._receive_task = asyncio.create_task(
            self._receive_loop(),
            name="deepgram-receiver",
        )

        self._keepalive_task = asyncio.create_task(
            self._keepalive_loop(),
            name="deepgram-keepalive",
        )

        logger.info(
            "Deepgram connection opened."
        )

    async def send_audio(
        self,
        pcm16_bytes: bytes,
    ) -> None:
        if not pcm16_bytes:
            return

        if not self._running:
            await self.start()

        ws = self._ws

        if ws is None:
            raise RuntimeError(
                "Deepgram WebSocket is not connected."
            )

        async with self._send_lock:
            await ws.send(
                pcm16_bytes
            )

    async def _send_json(
        self,
        payload: dict,
    ) -> None:
        ws = self._ws

        if ws is None:
            return

        async with self._send_lock:
            await ws.send(
                json.dumps(
                    payload
                )
            )

    async def _keepalive_loop(
        self,
    ) -> None:
        try:
            while self._running:
                await asyncio.sleep(5)

                if not self._running:
                    return

                await self._send_json(
                    {
                        "type":
                            "KeepAlive",
                    }
                )

        except asyncio.CancelledError:
            raise

        except Exception:
            if self._running:
                logger.exception(
                    "Deepgram keepalive failed."
                )

    async def _receive_loop(
        self,
    ) -> None:
        ws = self._ws

        if ws is None:
            return

        try:
            async for raw in ws:
                if not isinstance(
                    raw,
                    str,
                ):
                    continue

                try:
                    message = json.loads(
                        raw
                    )
                except json.JSONDecodeError:
                    continue

                if (
                    message.get("type")
                    != "Results"
                ):
                    continue

                channel = (
                    message.get("channel")
                    or {}
                )

                alternatives = (
                    channel.get("alternatives")
                    or []
                )

                if not alternatives:
                    continue

                best = (
                    alternatives[0]
                    or {}
                )

                text = str(
                    best.get(
                        "transcript",
                        "",
                    )
                ).strip()

                if not text:
                    continue

                confidence_raw = (
                    best.get(
                        "confidence"
                    )
                )

                try:
                    confidence = (
                        float(confidence_raw)
                        if confidence_raw is not None
                        else None
                    )
                except (
                    TypeError,
                    ValueError,
                ):
                    confidence = None

                await self._callback(
                    TranscriptEvent(
                        text=text,
                        is_final=bool(
                            message.get(
                                "is_final",
                                False,
                            )
                        ),
                        speech_final=bool(
                            message.get(
                                "speech_final",
                                False,
                            )
                        ),
                        confidence=confidence,
                        detected_language=majority_word_language(
                            best.get("words") or []
                        ),
                    )
                )

        except ConnectionClosed:
            if self._running:
                logger.warning(
                    "Deepgram connection closed."
                )

        except asyncio.CancelledError:
            raise

        except Exception:
            if self._running:
                logger.exception(
                    "Deepgram receiver failed."
                )

        finally:
            self._running = False

    async def stop(self) -> None:
        if (
            not self._running
            and self._ws is None
        ):
            return

        self._running = False

        ws = self._ws

        if ws is not None:
            try:
                async with self._send_lock:
                    await ws.send(
                        json.dumps(
                            {
                                "type":
                                    "CloseStream",
                            }
                        )
                    )
            except Exception:
                pass

        for task in (
            self._keepalive_task,
            self._receive_task,
        ):
            if (
                task is not None
                and not task.done()
            ):
                task.cancel()

        for task in (
            self._keepalive_task,
            self._receive_task,
        ):
            if task is None:
                continue

            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception:
                pass

        self._keepalive_task = None
        self._receive_task = None

        if ws is not None:
            try:
                await ws.close()
            except Exception:
                pass

        self._ws = None

        logger.info(
            "Deepgram stopped."
        )
