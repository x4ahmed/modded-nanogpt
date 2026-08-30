"""Shared dependency gates for the RTX-local-v1 attention tests."""

from __future__ import annotations

import importlib.util
import os
import unittest


MESSAGE = (
    "RTX_REQUIRE_GPU_TESTS=1 but CUDA is unavailable: this suite cannot certify "
    "attention or CE parity without a device"
)


def cuda_available() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return torch.cuda.is_available()


def gpu_tests_required() -> bool:
    return os.environ.get("RTX_REQUIRE_GPU_TESTS") == "1"


def requires_cuda(cls):
    """Skip a TestCase without CUDA, unless RTX_REQUIRE_GPU_TESTS=1 demands a real run."""
    if cuda_available():
        return cls
    if gpu_tests_required():
        class MissingCuda(unittest.TestCase):
            def test_cuda_is_required(self):
                self.fail(MESSAGE)

        MissingCuda.__name__ = cls.__name__
        MissingCuda.__qualname__ = getattr(cls, "__qualname__", cls.__name__)
        return MissingCuda
    return unittest.skip("CUDA is not available; set RTX_REQUIRE_GPU_TESTS=1 to require it")(cls)


def require_triton() -> None:
    """Require Triton only for the fused fullgraph test."""
    if importlib.util.find_spec("triton") is not None:
        return
    message = "Triton is unavailable; fused FlexAttention compilation was not tested"
    if gpu_tests_required():
        raise AssertionError(f"RTX_REQUIRE_GPU_TESTS=1 but {message}")
    raise unittest.SkipTest(message)
