from __future__ import annotations
import asyncio
import json
from voice.transport.websocket import BrowserTransport


class FakeWebSocket:
    def __init__(self):
        self.text = []
        self.binary = []
        self.accepted = False
    async def accept(self): self.accepted = True
    async def send_text(self, value): self.text.append(value)
    async def send_bytes(self, value): self.binary.append(value)
    async def close(self, code=1000): pass


def test_transport_json_and_audio_contract():
    async def run():
        ws = FakeWebSocket()
        transport = BrowserTransport(ws)
        await transport.accept()
        await transport.send_event({"type":"hello"})
        await transport.output.enqueue(b"\x00\x00\x00\x00" * 100)
        await transport.output.clear()
        assert ws.accepted
        assert len(ws.binary) == 1
        events = [json.loads(x) for x in ws.text]
        assert events[0]["type"] == "hello"
        assert events[-1]["type"] == "clear_audio"
    asyncio.run(run())
