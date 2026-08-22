from __future__ import annotations
import asyncio
from voice.turn_manager import TurnManager, TurnManagerConfig


def test_turn_finalizes_after_silence_and_settle():
    async def run():
        results = []
        manager = TurnManager(TurnManagerConfig(20, 20), results.append)
        manager.on_speech_start()
        manager.on_final_transcript("What is my balance?")
        manager.on_speech_end()
        await asyncio.sleep(0.08)
        assert len(results) == 1
        assert results[0].transcript == "What is my balance?"
        await manager.shutdown()
    asyncio.run(run())


def test_speech_resume_cancels_endpoint():
    async def run():
        results = []
        manager = TurnManager(TurnManagerConfig(30, 20), results.append)
        manager.on_speech_start()
        manager.on_partial_transcript("I lost")
        manager.on_speech_end()
        await asyncio.sleep(0.01)
        manager.on_speech_start()
        manager.on_final_transcript("I lost my job")
        await asyncio.sleep(0.05)
        assert results == []
        manager.on_speech_end()
        await asyncio.sleep(0.08)
        assert len(results) == 1
        assert results[0].transcript == "I lost my job"
        await manager.shutdown()
    asyncio.run(run())
