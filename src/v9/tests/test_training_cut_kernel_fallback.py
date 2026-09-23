from __future__ import annotations

from v9.hgt.training_cut import TrainingDeterminismMode, resolve_deterministic_kernel


def test_deterministic_kernel_fallback_order() -> None:
    assert resolve_deterministic_kernel(cuda_deterministic=True, deterministic_replacement=True, cpu_fallback=True) is TrainingDeterminismMode.DETERMINISTIC_CUDA
    assert resolve_deterministic_kernel(cuda_deterministic=False, deterministic_replacement=True, cpu_fallback=True) is TrainingDeterminismMode.DETERMINISTIC_REPLACEMENT
    assert resolve_deterministic_kernel(cuda_deterministic=False, deterministic_replacement=False, cpu_fallback=True) is TrainingDeterminismMode.DETERMINISTIC_CPU
    assert resolve_deterministic_kernel(cuda_deterministic=False, deterministic_replacement=False, cpu_fallback=False) is TrainingDeterminismMode.NONDETERMINISTIC_UNSUPPORTED
