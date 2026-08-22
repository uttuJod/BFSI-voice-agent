from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket
from fastapi.responses import FileResponse

from business.durable_executor import DurableWriteExecutor
from jobs import JobStore
from business import (
    BusinessDatabase,
    BusinessRepository,
    BusinessToolExecutor,
    VerificationRegistry,
)
from integration import (
    BFSIOrchestrator,
)

from integration.production_router import (
    ProductionBFSIRouter,
)
from integration.runtime_latency import (
    install_latency_optimizations,
)
from voice.localization import (
    ResponseLocalizer,
)
from voice.runtime import (
    VoiceSession,
)
from voice.stt.deepgram import (
    DeepgramStreamingSTT,
)
from voice.transport.websocket import (
    BrowserTransport,
)
from voice.tts.cartesia import (
    CartesiaStreamingTTS,
)


logger = logging.getLogger(__name__)


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)

WEB_ROOT = (
    PROJECT_ROOT
    / "web"
)


def _build_orchestrator(
) -> tuple[BFSIOrchestrator, VerificationRegistry]:
    db_path = (
        PROJECT_ROOT
        / "data"
        / "business.db"
    )

    db = BusinessDatabase(
        db_path
    )

    db.initialize()
    db.seed_demo_data()

    repository = (
        BusinessRepository(
            db
        )
    )

    verification_registry = (
        VerificationRegistry(
            db,
            verified_ttl_seconds=float(
                os.getenv(
                    "BFSI_VERIFICATION_TTL_SECONDS",
                    "600",
                )
            ),
            max_attempts=int(
                os.getenv(
                    "BFSI_VERIFICATION_MAX_ATTEMPTS",
                    "3",
                )
            ),
        )
    )

    tools = (
        BusinessToolExecutor(
            repository,
            verification_registry=(
                verification_registry
            ),
        )
    )

    # Guaranteed-delivery pipeline for downstream side effects of write
    # tools. The local record stays synchronous; delivery is a durable job.
    # Disable with JOBS_ENABLED=0 (the executor is then used directly).
    if os.getenv("JOBS_ENABLED", "1").strip() not in {"0", "false", "no"}:
        job_store = JobStore(
            os.getenv("JOBS_DB", "results/jobs.db"),
            lease_seconds=float(os.getenv("JOBS_LEASE_S", "30")),
        )
        tools = DurableWriteExecutor(
            tools,
            job_store,
            scope=os.getenv("APP_INSTANCE_ID", "app"),
        )
        logger.info(
            "Durable write pipeline enabled | db=%s",
            job_store.path,
        )

    router = ProductionBFSIRouter(
        base_url=os.getenv(
            "VLLM_BASE_URL",
            "http://127.0.0.1:8001/v1",
        ).strip(),
        model=os.getenv(
            "VLLM_MODEL",
            "bfsi-router",
        ).strip(),
        timeout_seconds=float(
            os.getenv(
                "VLLM_TIMEOUT_SECONDS",
                "30",
            )
        ),
    )

    health = router.health()

    logger.info(
        "vLLM router connected | models=%s | health_ms=%.1f",
        health.get("models"),
        float(health.get("latency_ms", 0.0)),
    )

    # Warm the vLLM + LoRA path during application startup so the
    # first real customer turn does not pay the cold-start penalty.
    warmup_started = asyncio.get_event_loop().time()
    router.route("What is my outstanding balance?")
    warmup_ms = (
        asyncio.get_event_loop().time()
        - warmup_started
    ) * 1000

    logger.info(
        "vLLM router warmup complete | warmup_ms=%.1f",
        warmup_ms,
    )

    orchestrator = BFSIOrchestrator(
        router=router,
        tool_executor=tools,
    )

    install_latency_optimizations(
        orchestrator,
        max_new_tokens=320,
        move_embeddings_cpu=(
            os.getenv(
                "RAG_EMBEDDINGS_DEVICE",
                "cpu",
            )
            .strip()
            .lower()
            == "cpu"
        ),
    )

    return (
        orchestrator,
        verification_registry,
    )


@asynccontextmanager
async def lifespan(
    app: FastAPI,
):
    load_dotenv()

    logger.info(
        "Initializing clean BFSI core..."
    )

    (
        app.state.orchestrator,
        app.state.verification_registry,
    ) = _build_orchestrator()

    app.state.localizer = (
        ResponseLocalizer(
            hindi_model_name=os.getenv(
                "HINDI_TRANSLATION_MODEL",
                "Helsinki-NLP/opus-mt-en-hi",
            ),
            device=os.getenv(
                "TRANSLATION_DEVICE",
                "cpu",
            ),
        )
    )

    app.state.inference_lock = (
        asyncio.Lock()
    )

    app.state.voice_session_lock = (
        asyncio.Lock()
    )

    logger.info(
        "Clean BFSI core ready."
    )

    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title=(
            "BFSI Voice Agent "
            "(Language Locked)"
        ),
        lifespan=lifespan,
    )

    @app.get("/")
    async def index():
        return FileResponse(
            WEB_ROOT
            / "index.html"
        )

    @app.get(
        "/pcm-worklet.js"
    )
    async def worklet():
        return FileResponse(
            WEB_ROOT
            / "pcm-worklet.js",
            media_type=(
                "application/javascript"
            ),
        )

    @app.get("/health")
    async def health():
        return {
            "ok": True,
            "livekit": False,
            "ollama": False,
            "output_languages": [
                "english",
                "hindi",
            ],
            "ai_core":
                "qwen35_vllm_rag_business_tools",
        }

    @app.websocket(
        "/ws/voice"
    )
    async def voice_socket(
        websocket: WebSocket,
    ):
        lock: asyncio.Lock = (
            app.state
            .voice_session_lock
        )

        if lock.locked():
            await websocket.accept()

            try:
                await websocket.send_json(
                    {
                        "type":
                            "error",
                        "message": (
                            "Another voice session "
                            "is already active."
                        ),
                    }
                )
            finally:
                try:
                    await websocket.close(
                        code=1013
                    )
                except Exception:
                    pass

            return

        async with lock:
            transport = (
                BrowserTransport(
                    websocket
                )
            )

            customer_id = os.getenv(
                "BFSI_CUSTOMER_ID",
                "CUST-1001",
            ).strip()

            deepgram_key = os.getenv(
                "DEEPGRAM_API_KEY",
                "",
            ).strip()

            cartesia_key = os.getenv(
                "CARTESIA_API_KEY",
                "",
            ).strip()

            cartesia_voice = os.getenv(
                "CARTESIA_VOICE_ID",
                "",
            ).strip()

            session_ref: dict = {}

            async def on_transcript(
                event,
            ) -> None:
                session = (
                    session_ref.get(
                        "session"
                    )
                )

                if session is not None:
                    await session.on_transcript(
                        event
                    )

            stt = DeepgramStreamingSTT(
                api_key=deepgram_key,
                transcript_callback=(
                    on_transcript
                ),
                model=os.getenv(
                    "DEEPGRAM_MODEL",
                    "nova-3",
                ),
                language_mode=(
                    "english"
                ),
            )

            tts = CartesiaStreamingTTS(
                api_key=cartesia_key,
                voice_id=cartesia_voice,
                model_id=os.getenv(
                    "CARTESIA_MODEL",
                    "sonic-3.5",
                ),
            )

            verification_registry = (
                app.state
                .verification_registry
            )

            verification_session_id = (
                verification_registry
                .create_session(
                    customer_id
                )
            )

            session = VoiceSession(
                transport=transport,
                orchestrator=(
                    app.state
                    .orchestrator
                ),
                stt=stt,
                tts=tts,
                localizer=(
                    app.state
                    .localizer
                ),
                customer_id=customer_id,
                inference_lock=(
                    app.state
                    .inference_lock
                ),
                verification_registry=(
                    verification_registry
                ),
                verification_session_id=(
                    verification_session_id
                ),
                barge_confirm_ms=int(
                    os.getenv(
                        "BARGE_CONFIRM_MS",
                        "120",
                    )
                ),
            )

            session_ref[
                "session"
            ] = session

            try:
                await session.run()
            finally:
                verification_registry.close_session(
                    verification_session_id
                )

    return app


app = create_app()
