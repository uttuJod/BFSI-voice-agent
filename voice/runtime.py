from __future__ import annotations

import asyncio
import json
import logging
import os
import time

from integration.orchestrator import (
    BFSIOrchestrator,
)
from business.verification import (
    VerificationRegistry,
)

from .localization import (
    ResponseLocalizer,
)
from .pre_orchestrator import (
    PreOrchestratorGuard,
)
from .idempotency import (
    SessionWriteIdempotency,
)
from .stt.deepgram import (
    DeepgramStreamingSTT,
    TranscriptEvent,
)
from .transport.websocket import (
    BrowserTransport,
)
from .tts.cartesia import (
    CartesiaStreamingTTS,
)
from .turn_manager import (
    TurnManager,
    TurnManagerConfig,
    TurnResult,
)
from .barge_metrics import BargeInMetrics
from .turn_metrics import TurnMetrics
from .language_detect import LanguageDetector
from .vad import (
    SileroStreamingVAD,
    VADEventType,
)


logger = logging.getLogger(__name__)


class VoiceSession:
    """
    Output language policy is chosen by the UI:

        auto    (default) reply in the language the caller used
                for the current utterance. Hindi or Hinglish input
                produces a Hindi reply; English input produces an
                English reply. Switching languages mid-session never
                resets conversation, verification or idempotency state.
        english always reply in English, whatever the caller spoke.
        hindi   always reply in Hindi, whatever the caller spoke.

    `language_policy` holds the UI selection.
    `output_language` holds the language resolved for the current turn
    and is the only value used for localization, TTS and UI events.

    Router/RAG/tools remain untouched by language handling.
    """

    VALID_LANGUAGES = {
        "auto",
        "english",
        "hindi",
    }

    def __init__(
        self,
        *,
        transport: BrowserTransport,
        orchestrator: BFSIOrchestrator,
        stt: DeepgramStreamingSTT,
        tts: CartesiaStreamingTTS,
        localizer: ResponseLocalizer,
        customer_id: str,
        inference_lock: asyncio.Lock,
        verification_registry: VerificationRegistry | None = None,
        verification_session_id: str | None = None,
        barge_confirm_ms: int = 120,
    ) -> None:
        self.transport = transport
        self.orchestrator = orchestrator
        self.stt = stt
        self.tts = tts
        self.localizer = localizer
        self.customer_id = customer_id
        self.inference_lock = inference_lock
        self.verification_registry = verification_registry
        self.verification_session_id = verification_session_id

        self.pre_orchestrator = (
            PreOrchestratorGuard()
        )

        self.write_idempotency = (
            SessionWriteIdempotency(
                ttl_seconds=60.0,
            )
        )

        self._inference_tasks: set[
            asyncio.Task
        ] = set()

        self.vad = SileroStreamingVAD()

        self.turn_manager = TurnManager(
            config=TurnManagerConfig(
                endpoint_silence_ms=int(
                    os.getenv(
                        "TURN_ENDPOINT_SILENCE_MS",
                        "550",
                    )
                ),
                asr_settle_ms=int(
                    os.getenv(
                        "TURN_ASR_SETTLE_MS",
                        "300",
                    )
                ),
            ),
            turn_callback=(
                self._on_turn_finalized
            ),
        )

        self._normal_turn_config = (
            self.turn_manager.config
        )

        #
        # Verification digits are commonly spoken with short pauses.
        # Only verification turns get this longer endpoint window.
        #
        self._verification_turn_config = TurnManagerConfig(
            endpoint_silence_ms=int(
                os.getenv(
                    "VERIFY_ENDPOINT_SILENCE_MS",
                    "1600",
                )
            ),
            asr_settle_ms=int(
                os.getenv(
                    "VERIFY_ASR_SETTLE_MS",
                    "900",
                )
            ),
        )

        self.turn_metrics = TurnMetrics()
        self._barge_counter = 0
        self._pending_barge_acks: dict[int, tuple[float | None, float]] = {}
        self.barge_metrics = BargeInMetrics()

        self.language_policy = "auto"
        self.output_language = "english"
        self.language_detector = LanguageDetector()
        self._last_detected_language: str | None = None

        self.session_active = False
        self.user_turn_open = False

        self.generation_id = 0

        self._current_task: (
            asyncio.Task[None] | None
        ) = None

        self._background_tasks: set[
            asyncio.Task
        ] = set()

        self._phase = "idle"

        self.barge_candidate_active = False
        self.barge_candidate_started_at = 0.0
        self.barge_candidate_speech_ended = False
        self._barge_expiry_task: (
            asyncio.Task[None] | None
        ) = None

        self.barge_candidate_ttl_s = 2.0

        self.barge_min_speech_ms = int(
            os.getenv("BARGE_MIN_SPEECH_MS", "250")
        )

        self.barge_confirm_s = max(
            0.0,
            barge_confirm_ms / 1000.0,
        )

        self._typed_turn_id = (
            1_000_000
        )

        # End-to-end latency marker:
        # user speech-end -> first assistant audio sent.
        self._last_speech_end_at: float | None = None

    async def run(self) -> None:
        await self.transport.accept()

        await self.transport.send_event(
            {
                "type": "ready",
                "customer_id":
                    self.customer_id,
                "language":
                    self.output_language,
            }
        )

        try:
            while True:
                message = (
                    await self.transport.receive()
                )

                if (
                    message.get("type")
                    == "websocket.disconnect"
                ):
                    break

                raw = message.get(
                    "bytes"
                )

                if raw is not None:
                    await self._handle_audio(
                        raw
                    )
                    continue

                text = message.get(
                    "text"
                )

                if text is not None:
                    await self._handle_control_text(
                        text
                    )

        finally:
            await self.shutdown()

    async def _handle_control_text(
        self,
        raw: str,
    ) -> None:
        try:
            message = json.loads(
                raw
            )
        except json.JSONDecodeError:
            return

        if not isinstance(
            message,
            dict,
        ):
            return

        message_type = str(
            message.get(
                "type",
                "",
            )
        )

        if (
            message_type
            == "start_session"
        ):
            requested_language = str(
                message.get(
                    "language",
                    "english",
                )
            ).strip().lower()

            if (
                requested_language
                not in self.VALID_LANGUAGES
            ):
                requested_language = (
                    "auto"
                )

            if self.stt.running:
                await self.stt.stop()

            self.language_policy = (
                requested_language
            )

            # Until the first utterance arrives, "auto" replies in English.
            self._apply_output_language(
                "english"
                if requested_language == "auto"
                else requested_language
            )

            # STT always recognises multi-language input; the mode is only
            # a recognition bias and never controls the answer language.
            self.stt.set_language_mode(
                "auto"
                if self.language_policy == "auto"
                else self.language_policy
            )

            self.session_active = True
            self.user_turn_open = False

            self.vad.reset()

            logger.info(
                "VOICE SESSION START | output_language=%s",
                self.output_language,
            )

            await self.transport.send_event(
                {
                    "type":
                        "session_state",
                    "active":
                        True,
                    "language":
                        self.output_language,
                }
            )

            await self._publish_state(
                "listening",
                (
                    "Listening · output="
                    f"{self.output_language}"
                ),
            )

            return

        if (
            message_type
            == "stop_session"
        ):
            self.session_active = False
            self.user_turn_open = False

            self._clear_barge_candidate()
            self.vad.reset()

            await self.transport.output.clear()
            await self.stt.stop()

            await self.transport.send_event(
                {
                    "type":
                        "session_state",
                    "active":
                        False,
                    "language":
                        self.output_language,
                }
            )

            return

        if (
            message_type
            == "barge_ack"
        ):
            self._record_barge_ack(message)
            return

        if (
            message_type
            == "text_query"
        ):
            query = str(
                message.get(
                    "text",
                    "",
                )
            ).strip()

            if not query:
                return

            self._typed_turn_id += 1

            await self.transport.send_event(
                {
                    "type":
                        "transcript",
                    "text":
                        query,
                    "final":
                        True,
                    "source":
                        "typed",
                }
            )

            self._schedule_response(
                TurnResult(
                    turn_id=(
                        self._typed_turn_id
                    ),
                    transcript=query,
                )
            )

    async def _handle_audio(
        self,
        pcm16: bytes,
    ) -> None:
        if not self.session_active:
            return

        if not self.stt.running:
            await self.stt.start()

            await self.transport.send_event(
                {
                    "type":
                        "stt_state",
                    "connected":
                        True,
                    "language":
                        self.output_language,
                }
            )

        await self.stt.send_audio(
            pcm16
        )

        for event in (
            self.vad.process_pcm16(
                pcm16
            )
        ):
            if (
                event.type
                is VADEventType.SPEECH_START
            ):
                await self._speech_start()

            elif (
                event.type
                is VADEventType.SPEECH_END
            ):
                await self._speech_end()

    async def _speech_start(
        self,
    ) -> None:
        logger.info(
            "========== SPEECH START =========="
        )

        if self._assistant_busy():
            self._clear_barge_candidate()

            self.barge_candidate_active = True

            self.barge_candidate_started_at = (
                time.perf_counter()
            )

            self.barge_candidate_speech_ended = False

            logger.info(
                "Barge-in candidate detected."
            )

            return

        self.user_turn_open = True

        self.turn_manager.on_speech_start()

    async def _speech_end(
        self,
    ) -> None:
        logger.info(
            "========== SPEECH END CANDIDATE =========="
        )

        if self.barge_candidate_active:
            voiced_ms = (
                time.perf_counter()
                - self.barge_candidate_started_at
            ) * 1000

            # Coughs, clicks and TTS bleed-through rarely exceed the
            # minimum voiced duration. Reject them before ASR even runs.
            if voiced_ms < self.barge_min_speech_ms:
                self.barge_metrics.record_rejected(
                    reason="too_short",
                    voiced_ms=voiced_ms,
                )
                logger.info(
                    "Barge-in candidate rejected | voiced_ms=%.0f < min=%d",
                    voiced_ms,
                    self.barge_min_speech_ms,
                )
                self._clear_barge_candidate()
                return

            self.barge_candidate_speech_ended = True
            self._schedule_barge_expiry()
            return

        if self.user_turn_open:
            self._last_speech_end_at = time.perf_counter()
            self.turn_manager.on_speech_end()

    async def on_transcript(
        self,
        event: TranscriptEvent,
    ) -> None:
        transcript = (
            event.text.strip()
        )

        if not transcript:
            return

        if event.is_final and event.detected_language:
            self._last_detected_language = (
                event.detected_language
            )

        await self.transport.send_event(
            {
                "type":
                    "transcript",
                "text":
                    transcript,
                "final":
                    event.is_final,
                "confidence":
                    event.confidence,
                "source":
                    "voice",
                "output_language":
                    self.output_language,
            }
        )

        if self.barge_candidate_active:
            if not self._barge_can_confirm(
                transcript=transcript,
                is_final=event.is_final,
            ):
                return

            speech_already_ended = (
                self.barge_candidate_speech_ended
            )

            await self._confirm_barge_in()

            self._feed_transcript(
                transcript,
                event.is_final,
            )

            if speech_already_ended:
                self.turn_manager.on_speech_end()

            return

        if self._assistant_busy():
            logger.info(
                "STALE ASR IGNORED WHILE ASSISTANT BUSY: %s",
                transcript,
            )
            return

        if not self.user_turn_open:
            logger.info(
                "STALE ASR IGNORED OUTSIDE ACTIVE USER TURN: %s",
                transcript,
            )
            return

        self._feed_transcript(
            transcript,
            event.is_final,
        )

    def _feed_transcript(
        self,
        transcript: str,
        is_final: bool,
    ) -> None:
        if is_final:
            logger.info(
                "FINAL   : %s",
                transcript,
            )

            self.turn_manager.on_final_transcript(
                transcript
            )

        else:
            logger.info(
                "PARTIAL : %s",
                transcript,
            )

            self.turn_manager.on_partial_transcript(
                transcript
            )

    def _schedule_barge_expiry(
        self,
    ) -> None:
        task = self._barge_expiry_task

        if (
            task is not None
            and not task.done()
        ):
            task.cancel()

        self._barge_expiry_task = (
            asyncio.create_task(
                self._expire_barge_candidate(),
                name=(
                    "barge-candidate-expiry"
                ),
            )
        )

    async def _expire_barge_candidate(
        self,
    ) -> None:
        try:
            await asyncio.sleep(
                self.barge_candidate_ttl_s
            )
        except asyncio.CancelledError:
            return

        if self.barge_candidate_active:
            logger.info(
                "Barge-in candidate expired without ASR confirmation."
            )
            self.barge_metrics.record_rejected(
                reason="no_asr_confirmation",
                voiced_ms=None,
            )

            self._clear_barge_candidate()

    def _clear_barge_candidate(
        self,
    ) -> None:
        self.barge_candidate_active = False
        self.barge_candidate_started_at = 0.0
        self.barge_candidate_speech_ended = False

        task = self._barge_expiry_task

        if (
            task is not None
            and not task.done()
            and task
            is not asyncio.current_task()
        ):
            task.cancel()

        self._barge_expiry_task = None

    def _barge_can_confirm(
        self,
        *,
        transcript: str,
        is_final: bool,
    ) -> bool:
        age = (
            time.perf_counter()
            - self.barge_candidate_started_at
        )

        if is_final:
            return True

        return (
            age >= self.barge_confirm_s
            and len(
                transcript.strip()
            ) >= 2
        )

    async def _confirm_barge_in(
        self,
    ) -> None:
        onset_at = self.barge_candidate_started_at

        self._clear_barge_candidate()

        self.generation_id += 1
        self._barge_counter += 1
        barge_id = self._barge_counter

        logger.info(
            "========== REAL BARGE-IN CONFIRMED =========="
        )

        started = (
            time.perf_counter()
        )

        await self.transport.output.clear()

        task = self._current_task

        if (
            task is not None
            and not task.done()
            and self._phase
            != "inference"
        ):
            task.cancel()

        self._current_task = None
        self._phase = "idle"

        self.user_turn_open = True
        self.turn_manager.on_speech_start()

        now = time.perf_counter()

        latency_ms = (now - started) * 1000

        # Time from VAD speech onset to the server issuing the cancel.
        # The browser reports its own stop time in `barge_ack` and the
        # two are summed into the PS3 "barge-in stop latency" metric.
        onset_to_cancel_ms = (
            (now - onset_at) * 1000
            if onset_at
            else None
        )

        self._pending_barge_acks[barge_id] = (
            onset_to_cancel_ms,
            now,
        )

        logger.info(
            "BARGE-IN AUDIO CANCEL = %.1f ms | onset_to_cancel_ms=%s",
            latency_ms,
            (
                f"{onset_to_cancel_ms:.1f}"
                if onset_to_cancel_ms is not None
                else "n/a"
            ),
        )

        await self.transport.send_event(
            {
                "type":
                    "barge_in",
                "confirmed":
                    True,
                "barge_id":
                    barge_id,
                "cancel_ms":
                    round(
                        latency_ms,
                        1,
                    ),
                "onset_to_cancel_ms":
                    (
                        round(onset_to_cancel_ms, 1)
                        if onset_to_cancel_ms is not None
                        else None
                    ),
            }
        )

    def _on_turn_finalized(
        self,
        result: TurnResult,
    ) -> None:
        self.user_turn_open = False
        self._schedule_response(
            result
        )

    def _schedule_response(
        self,
        result: TurnResult,
    ) -> None:
        # Cancel stale response tasks that have NOT entered router inference.
        # Active vLLM requests are allowed to finish; stale results are discarded
        # by generation_id before localization/TTS.
        current = asyncio.current_task()

        for old_task in list(
            self._background_tasks
        ):
            if (
                old_task is current
                or old_task.done()
                or old_task in self._inference_tasks
            ):
                continue

            old_task.cancel()

        self.generation_id += 1

        generation_id = (
            self.generation_id
        )

        task = asyncio.create_task(
            self._respond(
                result=result,
                generation_id=(
                    generation_id
                ),
            ),
            name=(
                "voice-response-"
                f"{generation_id}"
            ),
        )

        self._current_task = task

        self._background_tasks.add(
            task
        )

        task.add_done_callback(
            self._background_tasks.discard
        )

    def _orchestrator_stage_ms(self) -> dict[str, float]:
        tracker = getattr(self.orchestrator, "latency", None)
        if tracker is None or not hasattr(tracker, "last"):
            return {}
        try:
            return {
                f"{stage}_ms": value
                for stage, value in tracker.last().items()
            }
        except Exception:
            return {}

    def _record_barge_ack(
        self,
        message: dict,
    ) -> None:
        try:
            barge_id = int(message.get("barge_id"))
            client_stop_ms = float(message.get("client_stop_ms") or 0.0)
        except (TypeError, ValueError):
            return

        pending = self._pending_barge_acks.pop(barge_id, None)
        if pending is None:
            return

        onset_to_cancel_ms, sent_at = pending
        round_trip_ms = (time.perf_counter() - sent_at) * 1000

        sample = self.barge_metrics.record(
            onset_to_cancel_ms=onset_to_cancel_ms,
            server_to_client_one_way_ms=round_trip_ms / 2.0,
            client_stop_ms=client_stop_ms,
        )

        logger.info(
            "BARGE | stop_latency_ms=%.1f | onset_to_cancel_ms=%s | "
            "one_way_ms=%.1f | client_stop_ms=%.2f | n=%d | p50=%.1f | p95=%.1f",
            sample.stop_latency_ms,
            (
                f"{onset_to_cancel_ms:.1f}"
                if onset_to_cancel_ms is not None
                else "n/a"
            ),
            round_trip_ms / 2.0,
            client_stop_ms,
            self.barge_metrics.count,
            self.barge_metrics.p50,
            self.barge_metrics.p95,
        )

    def _apply_output_language(
        self,
        language: str,
    ) -> None:
        language = (
            "hindi"
            if language == "hindi"
            else "english"
        )
        self.output_language = language
        # TTS voice/language must match the localized text.
        self.tts.set_language_mode(language)

    def _resolve_output_language(
        self,
        transcript: str,
    ) -> str:
        """
        Decide the reply language for this turn.

        Fixed policies (english/hindi) always win.
        Under "auto" the detector combines the Deepgram per-word
        language hint for the last final transcript with a
        script/lexicon classifier on the text. Hinglish resolves to
        Hindi output, matching the evaluation expectation that a
        code-mixed utterance receives a Hindi reply.
        """
        if self.language_policy in {"english", "hindi"}:
            self._apply_output_language(self.language_policy)
            return self.output_language

        detection = self.language_detector.detect(
            transcript,
            stt_hint=self._last_detected_language,
        )

        self._last_detected_language = None

        previous = self.output_language
        self._apply_output_language(
            detection.output_language
        )

        logger.info(
            "LANGUAGE | policy=auto | detected=%s | output=%s | "
            "reason=%s",
            detection.detected,
            self.output_language,
            detection.reason,
        )

        if previous != self.output_language:
            logger.info(
                "LANGUAGE SWITCH | %s -> %s | session state retained",
                previous,
                self.output_language,
            )

        return self.output_language

    async def _speak_system_message(
        self,
        text: str,
        generation_id: int,
    ) -> None:
        final_text = await asyncio.to_thread(
            self.localizer.localize,
            text,
            self.output_language,
        )

        if generation_id != self.generation_id:
            return

        await self.transport.send_event(
            {
                "type": "assistant_text",
                "text": final_text,
                "language": self.output_language,
            }
        )

        self._phase = "tts"

        await self._publish_state(
            "speaking",
            "Speaking · verification",
        )

        async for audio in self.tts.synthesize(
            final_text
        ):
            if generation_id != self.generation_id:
                return

            await self.transport.output.enqueue(
                audio
            )

        await self.transport.output.wait_until_done()

        if generation_id == self.generation_id:
            self._phase = "idle"

            await self._publish_state(
                "listening",
                (
                    "Listening · output="
                    f"{self.output_language}"
                ),
            )

    async def _respond(
        self,
        *,
        result: TurnResult,
        generation_id: int,
    ) -> None:
        started = (
            time.perf_counter()
        )

        try:
            self._resolve_output_language(
                result.transcript
            )

            await self._publish_state(
                "thinking",
                "Processing request",
            )

            guarded = (
                self.pre_orchestrator.normalize(
                    result.transcript
                )
            )

            if guarded.actions:
                logger.info(
                    "PRE-GUARD | actions=%s | %s -> %s",
                    list(guarded.actions),
                    guarded.original,
                    guarded.normalized,
                )

            normalized_input = (
                guarded.normalized
            )

            #
            # Session identity-verification state machine.
            # A 4-digit reply while a challenge is pending is handled
            # locally and never sent to the LLM.
            #
            if (
                self.verification_registry is not None
                and self.verification_session_id
            ):
                verification_input = (
                    self.verification_registry
                    .consume_input(
                        self.verification_session_id,
                        self.customer_id,
                        normalized_input,
                    )
                )

                if verification_input.handled:
                    if (
                        verification_input.success
                        or verification_input.locked
                    ):
                        self.turn_manager.set_config(
                            self._normal_turn_config
                        )
                    else:
                        self.turn_manager.set_config(
                            self._verification_turn_config
                        )

                    await self._speak_system_message(
                        verification_input.message
                        or "Verification processed.",
                        generation_id,
                    )

                    if (
                        verification_input.success
                        and verification_input.resume_text
                        and generation_id
                        == self.generation_id
                    ):
                        #
                        # Resume the original account request after
                        # successful verification without asking the
                        # customer to repeat it.
                        #
                        resumed = TurnResult(
                            turn_id=result.turn_id,
                            transcript=(
                                verification_input
                                .resume_text
                            ),
                        )

                        await self._respond(
                            result=resumed,
                            generation_id=generation_id,
                        )

                    return

            duplicate = (
                self.write_idempotency.check(
                    normalized_input
                )
            )

            if duplicate is not None:
                logger.info(
                    "IDEMPOTENCY HIT | key=%s",
                    duplicate.key,
                )

                grounded_text = (
                    "That action was already recorded in this session, "
                    "so I did not create a duplicate."
                )

                final_text = (
                    await asyncio.to_thread(
                        self.localizer.localize,
                        grounded_text,
                        self.output_language,
                    )
                )

                await self.transport.send_event(
                    {
                        "type": "assistant_text",
                        "text": final_text,
                        "language": self.output_language,
                    }
                )

                self._phase = "tts"

                await self._publish_state(
                    "speaking",
                    "Speaking · duplicate protected",
                )

                async for audio in self.tts.synthesize(
                    final_text
                ):
                    if (
                        generation_id
                        != self.generation_id
                    ):
                        return

                    await self.transport.output.enqueue(
                        audio
                    )

                await self.transport.output.wait_until_done()

                if (
                    generation_id
                    == self.generation_id
                ):
                    await self._publish_state(
                        "listening",
                        (
                            "Listening · output="
                            f"{self.output_language}"
                        ),
                    )

                return

            logger.info(
                "VOICE → ORCHESTRATOR | output_language=%s | %s",
                self.output_language,
                normalized_input,
            )

            lock_started = (
                time.perf_counter()
            )

            current_task = (
                asyncio.current_task()
            )

            async with self.inference_lock:
                lock_wait_ms = (
                    time.perf_counter()
                    - lock_started
                ) * 1000

                logger.info(
                    "PERF | inference_lock_wait_ms=%.1f",
                    lock_wait_ms,
                )

                if current_task is not None:
                    self._inference_tasks.add(
                        current_task
                    )

                self._phase = "inference"

                orchestrator_started = time.perf_counter()

                try:
                    if (
                        self.verification_registry is not None
                        and self.verification_session_id
                    ):
                        with self.verification_registry.bind(
                            self.verification_session_id
                        ):
                            ai_result = (
                                await self.orchestrator.handle(
                                    user_text=(
                                        normalized_input
                                    ),
                                    customer_id=(
                                        self.customer_id
                                    ),
                                )
                            )
                    else:
                        ai_result = (
                            await self.orchestrator.handle(
                                user_text=(
                                    normalized_input
                                ),
                                customer_id=(
                                    self.customer_id
                                ),
                            )
                        )
                finally:
                    if current_task is not None:
                        self._inference_tasks.discard(
                            current_task
                        )

            orchestrator_ms = (
                time.perf_counter()
                - orchestrator_started
            ) * 1000

            self._phase = "localizing"

            if (
                generation_id
                != self.generation_id
            ):
                logger.info(
                    "Discarding stale AI result | generation_id=%d",
                    generation_id,
                )
                return

            if (
                self.verification_registry is not None
                and self.verification_session_id
                and ai_result.tool_error_code
                == "IDENTITY_VERIFICATION_REQUIRED"
            ):
                challenge = (
                    self.verification_registry.begin(
                        self.verification_session_id,
                        self.customer_id,
                        pending_request=normalized_input,
                    )
                )

                self.turn_manager.set_config(
                    self._verification_turn_config
                )

                await self._speak_system_message(
                    challenge,
                    generation_id,
                )

                return

            grounded_text = str(
                ai_result.final_response
                or ""
            ).strip()

            logger.info(
                "GROUNDED RESPONSE | %s",
                grounded_text,
            )

            if (
                ai_result.tool_executed
                and ai_result.tool_success
            ):
                self.write_idempotency.remember_success(
                    text=normalized_input,
                    tool_name=ai_result.router.tool,
                    grounded_response=grounded_text,
                )

            localize_started = (
                time.perf_counter()
            )

            final_text = (
                await asyncio.to_thread(
                    self.localizer.localize,
                    grounded_text,
                    self.output_language,
                )
            )

            localization_ms = (
                time.perf_counter()
                - localize_started
            ) * 1000

            if (
                generation_id
                != self.generation_id
            ):
                return

            elapsed_ms = (
                time.perf_counter()
                - started
            ) * 1000

            trace = {
                "output_language":
                    self.output_language,

                "intent":
                    ai_result.router.intent,

                "confidence":
                    ai_result.router.confidence,

                "requires_rag":
                    ai_result.router.requires_rag,

                "requires_tool":
                    ai_result.router.requires_tool,

                "tool":
                    ai_result.router.tool,

                "arguments":
                    ai_result.router.arguments,

                "source":
                    ai_result.source,

                "rag_used":
                    ai_result.rag_used,

                "rag_verdict":
                    ai_result.rag_verdict,

                "rag_citations":
                    ai_result.rag_citations,

                "tool_executed":
                    ai_result.tool_executed,

                "tool_success":
                    ai_result.tool_success,

                "tool_status":
                    ai_result.tool_status,

                "tool_result":
                    ai_result.tool_result,

                "guard_actions":
                    ai_result.guard_actions,

                "inference_lock_wait_ms":
                    round(
                        lock_wait_ms,
                        1,
                    ),

                "orchestrator_ms":
                    round(
                        orchestrator_ms,
                        1,
                    ),

                "localization_ms":
                    round(
                        localization_ms,
                        1,
                    ),

                "total_ai_ms":
                    round(
                        elapsed_ms,
                        1,
                    ),
            }

            logger.info(
                "AI TRACE | %s",
                trace,
            )

            logger.info(
                "FINAL RESPONSE | language=%s | %s",
                self.output_language,
                final_text,
            )

            await self.transport.send_event(
                {
                    "type":
                        "ai_trace",
                    "trace":
                        trace,
                }
            )

            await self.transport.send_event(
                {
                    "type":
                        "assistant_text",
                    "text":
                        final_text,
                    "language":
                        self.output_language,
                }
            )

            if not final_text:
                return

            self._phase = "tts"

            await self._publish_state(
                "speaking",
                (
                    "Speaking · "
                    f"{self.output_language}"
                ),
            )

            tts_started = time.perf_counter()
            first_audio_sent = False

            async for audio in (
                self.tts.synthesize(
                    final_text
                )
            ):
                if (
                    generation_id
                    != self.generation_id
                ):
                    return

                await self.transport.output.enqueue(
                    audio
                )

                if not first_audio_sent:
                    first_audio_sent = True
                    first_audio_at = time.perf_counter()

                    tts_first_audio_ms = (
                        first_audio_at
                        - tts_started
                    ) * 1000

                    speech_end_to_first_audio_ms = None

                    if self._last_speech_end_at is not None:
                        speech_end_to_first_audio_ms = (
                            first_audio_at
                            - self._last_speech_end_at
                        ) * 1000

                    logger.info(
                        "PERF | orchestrator_ms=%.1f | "
                        "localization_ms=%.1f | "
                        "tts_first_audio_ms=%.1f | "
                        "speech_end_to_first_audio_ms=%s",
                        orchestrator_ms,
                        localization_ms,
                        tts_first_audio_ms,
                        (
                            f"{speech_end_to_first_audio_ms:.1f}"
                            if speech_end_to_first_audio_ms is not None
                            else "n/a"
                        ),
                    )

                    self.turn_metrics.record(
                        generation_id=generation_id,
                        output_language=self.output_language,
                        intent=ai_result.router.intent,
                        rag_used=bool(ai_result.rag_used),
                        tool=ai_result.router.tool,
                        stages={
                            **self._orchestrator_stage_ms(),
                            "orchestrator_ms": orchestrator_ms,
                            "localization_ms": localization_ms,
                            "tts_first_audio_ms": tts_first_audio_ms,
                            "speech_end_to_first_audio_ms": (
                                speech_end_to_first_audio_ms
                            ),
                        },
                    )

                    await self.transport.send_event(
                        {
                            "type": "voice_latency",
                            "generation_id": generation_id,
                            "orchestrator_ms": round(
                                orchestrator_ms,
                                1,
                            ),
                            "localization_ms": round(
                                localization_ms,
                                1,
                            ),
                            "tts_first_audio_ms": round(
                                tts_first_audio_ms,
                                1,
                            ),
                            "speech_end_to_first_audio_ms": (
                                round(
                                    speech_end_to_first_audio_ms,
                                    1,
                                )
                                if speech_end_to_first_audio_ms
                                is not None
                                else None
                            ),
                        }
                    )

            await self.transport.output.wait_until_done()

            if (
                generation_id
                != self.generation_id
            ):
                return

            await self._publish_state(
                "listening",
                (
                    "Listening · output="
                    f"{self.output_language}"
                ),
            )

            self._last_speech_end_at = None

        except asyncio.CancelledError:
            logger.info(
                "Voice response cancelled | generation_id=%d",
                generation_id,
            )
            raise

        except Exception:
            logger.exception(
                "Voice response pipeline failed."
            )

        finally:
            if (
                generation_id
                == self.generation_id
            ):
                self._phase = "idle"

            if (
                self._current_task
                is asyncio.current_task()
            ):
                self._current_task = None

    def _assistant_busy(
        self,
    ) -> bool:
        return (
            (
                self._current_task
                is not None
                and not self._current_task.done()
            )
            or self.transport.output.is_playing
        )

    async def _publish_state(
        self,
        state: str,
        detail: str,
    ) -> None:
        await self.transport.send_event(
            {
                "type":
                    "agent_state",
                "state":
                    state,
                "detail":
                    detail,
                "language":
                    self.output_language,
            }
        )

    async def shutdown(
        self,
    ) -> None:
        self.session_active = False
        self.user_turn_open = False

        self._clear_barge_candidate()

        await self.transport.output.clear()
        await self.turn_manager.shutdown()

        try:
            await self.stt.stop()
        except Exception:
            pass

        try:
            await self.tts.close()
        except Exception:
            pass

        await asyncio.sleep(0)
