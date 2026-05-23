from __future__ import annotations

from pathlib import Path

import pytest

from adaos.services.testing.bootstrap import bootstrap_test_ctx


@pytest.fixture(autouse=True)
def _skill_test_context():
    skill_dir = Path(__file__).resolve().parents[1]
    handle = bootstrap_test_ctx(
        skill_name=skill_dir.name,
        skill_slot_dir=skill_dir,
        secrets={},
    )
    try:
        yield handle.ctx
    finally:
        handle.teardown()
