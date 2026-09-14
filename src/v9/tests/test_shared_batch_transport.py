from v9.environments.synthetic_symbolic import SyntheticSymbolicConfig, SyntheticSymbolicEnvironment
from v9.runtime.shared_batch_transport import consume_shared_batch, publish_shared_batch


def test_shared_batch_transport_round_trip() -> None:
    value = {"rows": tuple((index, f"token-{index % 3}") for index in range(512))}
    descriptor = publish_shared_batch(value, start_sequence=1, end_sequence=512, rows=512)
    restored, decode_ms = consume_shared_batch(descriptor)
    assert restored == value
    assert descriptor.size > 0
    assert descriptor.rows == 512
    assert descriptor.encode_ms >= 0.0
    assert decode_ms >= 0.0


def test_synthetic_snapshot_restore_is_exact() -> None:
    env = SyntheticSymbolicEnvironment(
        SyntheticSymbolicConfig(seed=7, shuffled=True, horizon=8)
    )
    env.step(1)
    state = env.capture_state()
    first = (env.step(0), env.optional_symbol_stream(), env.boundary_event())
    env.restore_state(state)
    second = (env.step(0), env.optional_symbol_stream(), env.boundary_event())
    assert second == first
