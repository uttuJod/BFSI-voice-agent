from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from enum import Enum, auto
from typing import Callable


logger = logging.getLogger(__name__)


class TurnState(Enum):
    LISTENING = auto()
    ENDPOINTING = auto()


@dataclass(frozen=True, slots=True)
class TurnResult:
    turn_id: int
    transcript: str


TurnCallback = Callable[[TurnResult], None]


@dataclass(frozen=True, slots=True)
class TurnManagerConfig:
    endpoint_silence_ms: int = 900
    asr_settle_ms: int = 500


class TurnManager:
    def __init__(
        self,
        config: TurnManagerConfig,
        turn_callback: TurnCallback,
    ) -> None:
        self._config = config
        self._turn_callback = turn_callback

        self._state = TurnState.LISTENING
        self._turn_id = 0

        self._final_segments: list[str] = []
        self._latest_partial = ""

        self._speech_active = False

        self._endpoint_task: asyncio.Task | None = None
        self._asr_settle_task: asyncio.Task | None = None

        self._silence_timeout_complete = False
        self._last_final_version = 0

    @property
    def state(self) -> TurnState:
        return self._state

    @property
    def current_turn_id(self) -> int:
        return self._turn_id

    @property
    def config(
        self,
    ) -> TurnManagerConfig:
        return self._config

    def set_config(
        self,
        config: TurnManagerConfig,
    ) -> None:
        self._config = config

        logger.info(
            "TurnManager config updated | "
            "endpoint_silence_ms=%d | asr_settle_ms=%d",
            config.endpoint_silence_ms,
            config.asr_settle_ms,
        )

        if (
            self._state is TurnState.ENDPOINTING
            and not self._speech_active
            and not self._silence_timeout_complete
        ):
            self._start_endpoint_timer()


    def on_speech_start(self) -> None:
        logger.info(
            "TurnManager: speech start | turn_id=%d",
            self._turn_id,
        )

        self._speech_active = True
        self._state = TurnState.LISTENING
        self._silence_timeout_complete = False

        self._cancel_endpoint_timer()
        self._cancel_asr_settle_timer()

    def on_speech_end(self) -> None:
        logger.info(
            "TurnManager: speech end candidate | turn_id=%d",
            self._turn_id,
        )

        self._speech_active = False
        self._state = TurnState.ENDPOINTING
        self._silence_timeout_complete = False

        self._start_endpoint_timer()

    def on_partial_transcript(
        self,
        transcript: str,
    ) -> None:
        transcript = transcript.strip()

        if not transcript:
            return

        self._latest_partial = transcript

        logger.info(
            "TurnManager partial | turn_id=%d | %s",
            self._turn_id,
            transcript,
        )

        if (
            self._state is TurnState.ENDPOINTING
            and self._silence_timeout_complete
        ):
            self._restart_asr_settle_timer()

    def on_final_transcript(
        self,
        transcript: str,
    ) -> None:
        transcript = transcript.strip()

        if not transcript:
            return

        self._final_segments.append(transcript)
        self._latest_partial = ""

        self._last_final_version += 1

        logger.info(
            "TurnManager final segment | turn_id=%d | %s",
            self._turn_id,
            transcript,
        )

        if (
            self._state is TurnState.ENDPOINTING
            and self._silence_timeout_complete
        ):
            self._restart_asr_settle_timer()

    def _start_endpoint_timer(self) -> None:
        self._cancel_endpoint_timer()

        self._endpoint_task = asyncio.create_task(
            self._endpoint_after_silence(),
            name=f"endpoint-turn-{self._turn_id}",
        )

    async def _endpoint_after_silence(self) -> None:
        try:
            await asyncio.sleep(
                self._config.endpoint_silence_ms / 1000
            )

            if self._speech_active:
                return

            logger.info(
                "Silence timeout complete | turn_id=%d",
                self._turn_id,
            )

            self._silence_timeout_complete = True

            self._restart_asr_settle_timer()

        except asyncio.CancelledError:
            logger.info(
                "Endpoint timer cancelled | turn_id=%d",
                self._turn_id,
            )
            raise

    def _restart_asr_settle_timer(self) -> None:
        self._cancel_asr_settle_timer()

        expected_turn_id = self._turn_id
        expected_final_version = self._last_final_version

        self._asr_settle_task = asyncio.create_task(
            self._wait_for_asr_settle(
                expected_turn_id=expected_turn_id,
                expected_final_version=expected_final_version,
            ),
            name=f"asr-settle-turn-{self._turn_id}",
        )

    async def _wait_for_asr_settle(
        self,
        expected_turn_id: int,
        expected_final_version: int,
    ) -> None:
        try:
            await asyncio.sleep(
                self._config.asr_settle_ms / 1000
            )

            if self._speech_active:
                return

            if expected_turn_id != self._turn_id:
                return

            if expected_final_version != self._last_final_version:
                return

            self._finalize_turn()

        except asyncio.CancelledError:
            raise

    def _cancel_endpoint_timer(self) -> None:
        if self._endpoint_task is None:
            return

        if not self._endpoint_task.done():
            self._endpoint_task.cancel()

        self._endpoint_task = None

    def _cancel_asr_settle_timer(self) -> None:
        if self._asr_settle_task is None:
            return

        if not self._asr_settle_task.done():
            self._asr_settle_task.cancel()

        self._asr_settle_task = None

    def _build_transcript(self) -> str:
        final_text = " ".join(
            segment.strip()
            for segment in self._final_segments
            if segment.strip()
        ).strip()

        partial_text = self._latest_partial.strip()

        if final_text and partial_text:
            if partial_text.startswith(final_text):
                return partial_text

            return f"{final_text} {partial_text}".strip()

        if final_text:
            return final_text

        return partial_text

    def _finalize_turn(self) -> None:
        transcript = self._build_transcript()

        if not transcript:
            logger.info(
                "Ignoring empty turn | turn_id=%d",
                self._turn_id,
            )

            self._reset_for_next_turn()
            return

        result = TurnResult(
            turn_id=self._turn_id,
            transcript=transcript,
        )

        logger.info(
            "========== TURN FINALIZED =========="
        )

        logger.info(
            "turn_id=%d transcript=%s",
            result.turn_id,
            result.transcript,
        )

        self._turn_callback(result)

        self._reset_for_next_turn()

    def _reset_for_next_turn(self) -> None:
        self._cancel_endpoint_timer()
        self._cancel_asr_settle_timer()

        self._turn_id += 1

        self._final_segments.clear()
        self._latest_partial = ""

        self._speech_active = False
        self._state = TurnState.LISTENING

        self._silence_timeout_complete = False
        self._last_final_version = 0

    async def shutdown(self) -> None:
        self._cancel_endpoint_timer()
        self._cancel_asr_settle_timer()

        await asyncio.sleep(0)