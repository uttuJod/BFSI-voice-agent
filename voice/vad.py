from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum

import numpy as np
import torch
from silero_vad import VADIterator, load_silero_vad

logger = logging.getLogger(__name__)


class VADEventType(str, Enum):
    SPEECH_START = "speech_start"
    SPEECH_END = "speech_end"


@dataclass(frozen=True, slots=True)
class VADEvent:
    type: VADEventType


class SileroStreamingVAD:
    WINDOW_SAMPLES = 512

    def __init__(self, sample_rate: int = 16_000, threshold: float = 0.50, min_silence_ms: int = 250) -> None:
        if sample_rate != 16_000:
            raise ValueError("Voice runtime expects 16 kHz microphone PCM.")
        logger.info("Loading Silero VAD model.")
        self._model = load_silero_vad()
        self._iterator = VADIterator(
            self._model,
            threshold=threshold,
            sampling_rate=sample_rate,
            min_silence_duration_ms=min_silence_ms,
            speech_pad_ms=30,
        )
        self._buffer = np.empty(0, dtype=np.float32)

    def reset(self) -> None:
        self._buffer = np.empty(0, dtype=np.float32)
        self._iterator.reset_states()

    def process_pcm16(self, pcm_bytes: bytes) -> list[VADEvent]:
        usable = len(pcm_bytes) - (len(pcm_bytes) % 2)
        if usable <= 0:
            return []
        pcm16 = np.frombuffer(pcm_bytes[:usable], dtype="<i2")
        samples = pcm16.astype(np.float32) / 32768.0
        self._buffer = np.concatenate((self._buffer, samples))
        events: list[VADEvent] = []
        while self._buffer.size >= self.WINDOW_SAMPLES:
            chunk = self._buffer[: self.WINDOW_SAMPLES]
            self._buffer = self._buffer[self.WINDOW_SAMPLES :]
            result = self._iterator(torch.from_numpy(chunk.copy()), return_seconds=False)
            if not result:
                continue
            if "start" in result:
                events.append(VADEvent(VADEventType.SPEECH_START))
            if "end" in result:
                events.append(VADEvent(VADEventType.SPEECH_END))
        return events
