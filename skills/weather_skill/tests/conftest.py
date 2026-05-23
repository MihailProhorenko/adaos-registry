# src/adaos/skills/weather_skill/tests/conftest.py
import shutil
from pathlib import Path
import pytest

from adaos.services.testing.bootstrap import bootstrap_test_ctx


@pytest.fixture(autouse=True)
def _skill_test_context(tmp_path, monkeypatch):
    skill_dir = Path(__file__).resolve().parents[1]
    temp_slot = tmp_path / "skill-slot"
    temp_slot.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("ADAOS_BASE_DIR", str(tmp_path / ".adaos"))
    handle = bootstrap_test_ctx(
        skill_name=skill_dir.name,
        skill_slot_dir=temp_slot,
        secrets={},
    )
    try:
        yield handle.ctx
    finally:
        handle.teardown()


@pytest.fixture
def skill_root(tmp_path: Path) -> Path:
    """
    Копируем minimal prep внутрь tmp, чтобы тесты не трогали рабочую .adaos.
    """
    here = Path(__file__).resolve().parents[1]  # weather_skill/
    tmp_skill = tmp_path / "weather_skill"
    shutil.copytree(here, tmp_skill, dirs_exist_ok=True)
    return tmp_skill
