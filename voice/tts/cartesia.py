from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator

from cartesia import AsyncCartesia

from ..speech_normalizer import (
    SpeechTextNormalizer,
)


logger = logging.getLogger(__name__)


class CartesiaStreamingTTS:
    """
    Cartesia TTS adapter.

    The text shown in the UI remains unchanged.
    A TTS-only normalizer converts:
      - INR digits into natural Indian-number speech
      - Hindi first-person assistant grammar to feminine form
    """

    VALID_LANGUAGE_MODES = {
        "auto",
        "english",
        "hindi",
        "hinglish",
    }

    def __init__(
        self,
        *,
        api_key: str,
        voice_id: str,
        model_id: str = "sonic-3.5",
        sample_rate: int = 44_100,
    ) -> None:
        if not api_key.strip():
            raise RuntimeError(
                "CARTESIA_API_KEY is empty."
            )

        if not voice_id.strip():
            raise RuntimeError(
                "CARTESIA_VOICE_ID is empty."
            )

        self._client = AsyncCartesia(
            api_key=api_key.strip(),
        )

        self._voice_id = voice_id.strip()
        self._model_id = model_id
        self._sample_rate = sample_rate
        self._language_mode = "auto"

        self._speech_normalizer = (
            SpeechTextNormalizer()
        )

    @property
    def language_mode(self) -> str:
        return self._language_mode

    def set_language_mode(
        self,
        mode: str,
    ) -> str:
        mode = str(
            mode
            or "auto"
        ).strip().lower()

        if (
            mode
            not in self.VALID_LANGUAGE_MODES
        ):
            raise ValueError(
                "Language must be auto, english, hindi, or hinglish."
            )

        self._language_mode = mode
        return mode

    def _language_for_text(
        self,
        text: str,
    ) -> str:
        if self._language_mode == "english":
            return "en"

        if self._language_mode in {
            "hindi",
            "hinglish",
        }:
            return "hi"

        if any(
            "\u0900" <= char <= "\u097f"
            for char in text
        ):
            return "hi"

        return "en"

    async def synthesize(
        self,
        text: str,
    ) -> AsyncIterator[bytes]:
        display_text = str(
            text
            or ""
        ).strip()

        if not display_text:
            return

        language = (
            self._language_for_text(
                display_text
            )
        )

        normalizer_language = (
            "hindi"
            if language == "hi"
            else "english"
        )

        speech_text = (
            self._speech_normalizer.normalize(
                display_text,
                normalizer_language,
            )
        )

        logger.info(
            "TTS REQUEST | mode=%s language=%s | %s",
            self._language_mode,
            language,
            display_text,
        )

        if speech_text != display_text:
            logger.info(
                "TTS SPOKEN TEXT | %s",
                speech_text,
            )

        started = time.perf_counter()
        first_audio = True

        async with (
            self._client.tts
            .websocket_connect()
        ) as ws:
            ctx = ws.context(
                model_id=self._model_id,

                voice={
                    "mode": "id",
                    "id": self._voice_id,
                },

                output_format={
                    "container": "raw",
                    "encoding": "pcm_f32le",
                    "sample_rate":
                        self._sample_rate,
                },

                language=language,
            )

            await ctx.push(
                speech_text
            )

            await ctx.no_more_inputs()

            async for response in (
                ctx.receive()
            ):
                if (
                    response.type
                    == "chunk"
                    and response.audio
                ):
                    if first_audio:
                        first_audio = False

                        logger.info(
                            "TTS TTFB = %.1f ms",
                            (
                                time.perf_counter()
                                - started
                            )
                            * 1000,
                        )

                    yield response.audio

                elif (
                    response.type
                    == "error"
                ):
                    message = (
                        getattr(
                            response,
                            "message",
                            None,
                        )
                        or getattr(
                            response,
                            "title",
                            None,
                        )
                        or "Cartesia TTS error."
                    )

                    raise RuntimeError(
                        str(message)
                    )

    async def close(self) -> None:
        close = getattr(
            self._client,
            "close",
            None,
        )

        if close is None:
            return

        result = close()

        if hasattr(
            result,
            "__await__",
        ):
            await result
