from __future__ import annotations

from integration.runtime_latency import (
    SafeLatencyController,
    install_latency_optimizations,
)


def test_safe_controller_does_not_define_cancel_inference():
    assert not hasattr(
        SafeLatencyController,
        "cancel",
    )


def test_installer_signature_exists():
    assert callable(
        install_latency_optimizations
    )
