from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_module():
    path = Path(__file__).resolve().parents[1] / "handlers" / "main.py"
    spec = importlib.util.spec_from_file_location("neuro_lite_main", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_neuro_lite_detects_core_smoke_intents():
    module = _load_module()
    runtime = module.NeuroLiteRuntime()

    timer = runtime.detect("поставь таймер на 10 минут")
    assert timer["accepted"] is True
    assert timer["top_intent"] == "voice.timer.start"
    assert timer["slots"]["duration"] == "10 минут"

    marketplace = runtime.detect("открой маркетплейс")
    assert marketplace["accepted"] is True
    assert marketplace["top_intent"] == "desktop.open_marketplace"


def test_neuro_lite_separates_weather_task_from_open_weather_ui():
    module = _load_module()
    runtime = module.NeuroLiteRuntime()

    weather_task = runtime.detect("какая погода в Москве")
    assert weather_task["accepted"] is True
    assert weather_task["top_intent"] == "weather.show"
    assert weather_task["slots"]["city"] == "москве"

    open_weather = runtime.detect("открой погоду")
    assert open_weather["accepted"] is True
    assert open_weather["top_intent"] == "desktop.open_weather"
