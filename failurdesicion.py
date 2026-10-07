"""AI3 — prescriptive-оптимизатор. НЕ обучает модели, а загружает готовые:
    failure_binary_model.joblib  (AI1, python failureai.py)
    failure_type_model.joblib    (AI2, python failurecause.py)

Для frontend:
    analyze_machine(machine)  - один станок
    analyze_batch(machines)   - много станков (CSV), без тяжёлой оптимизации

Запуск:
    python failurdesicion.py                    - демо в консоли
    uvicorn failurdesicion:app --reload --port 8000   - HTTP API для сайта
Фронт: fetch("http://localhost:8000/analyze", {method:"POST",
       headers:{"Content-Type":"application/json"}, body: JSON.stringify({...})})
"""
from __future__ import annotations

import os
from functools import lru_cache
from typing import List, Literal

import numpy as np
import pandas as pd

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from failureai import (
    FEATURES,
    add_engineered_features,
    encode_type,
    load_model as _load_binary_model,
    predict_failure_risk,
)
from failurecause import predict_failure as predict_failure_causes

REQUIRED_KEYS = ("type_", "air_k", "proc_k", "rpm", "torque", "wear")

# Допустимые входы (диапазон датасета AI4I 2020 с запасом)
LIMITS = {
    "air_k": (290.0, 310.0),
    "proc_k": (300.0, 320.0),
    "rpm": (500.0, 4000.0),
    "torque": (0.0, 100.0),
    "wear": (0.0, 300.0),
}
# Верхняя граница температуры процесса в обучающих данных: выше модель только экстраполирует
PROC_MAX_K = 314.0
MIN_GAIN = 0.01   # улучшение меньше 1 п.п. не считаем рекомендацией

CAUSE_LABELS = {
    "TWF": "Износ инструмента",
    "HDF": "Недостаточный отвод тепла",
    "PWF": "Сбой по мощности",
    "OSF": "Перегрузка (overstrain)",
    "RNF": "Случайный отказ",
}


class MachineValidationError(ValueError):
    """Ошибка входных данных. .errors = {поле: сообщение} - удобно вернуть фронту как 422."""

    def __init__(self, errors: dict):
        super().__init__("; ".join(f"{k}: {v}" for k, v in errors.items()))
        self.errors = errors


def validate_machine(machine: dict) -> dict:
    """Приводит вход к чистому виду: принимает 'type' и 'type_', строки -> float, проверяет диапазоны."""
    m = dict(machine)
    if "type_" not in m and "type" in m:
        m["type_"] = m["type"]

    errors = {k: "обязательное поле" for k in REQUIRED_KEYS if m.get(k) is None}
    clean = {}

    if "type_" not in errors:
        t = str(m["type_"]).strip().upper()
        if t in ("L", "M", "H"):
            clean["type_"] = t
        else:
            errors["type_"] = "допустимо L, M или H"

    for key, (lo, hi) in LIMITS.items():
        if key in errors:
            continue
        try:
            v = float(m[key])
        except (TypeError, ValueError):
            errors[key] = "должно быть числом"
            continue
        if not np.isfinite(v) or not lo <= v <= hi:
            errors[key] = f"вне диапазона {lo}..{hi}"
        else:
            clean[key] = v

    if "rpm" not in errors and clean.get("rpm", 1) <= 0:
        errors["rpm"] = "должно быть больше 0"
    if errors:
        raise MachineValidationError(errors)
    return clean


@lru_cache(maxsize=1)
def _binary_model():
    """Модель AI1 грузится один раз на процесс, а не при каждом запросе."""
    return _load_binary_model()["model"]


def _risk_level(p: float) -> str:
    return "low" if p < 0.25 else "medium" if p < 0.6 else "high"


# ==========================================
# ОПТИМИЗАЦИЯ РЕЖИМА (GRID SEARCH)
# ==========================================
def _find_best_settings(data: dict, target: float) -> dict:
    """Ищет режим с риском <= target и минимальным отклонением от текущего.
    Если цель недостижима - берёт режим с минимальным риском."""
    air = data["air_k"]

    rpm_grid = np.arange(1200, 2800, 50.0)
    torque_grid = np.arange(15.0, 75.0, 2.5)
    proc_grid = np.arange(air + 8.0, air + 15.0, 1.0)
    proc_grid = proc_grid[proc_grid <= PROC_MAX_K]         # не выходим за обучающие данные
    proc_grid = np.unique(np.append(proc_grid, data["proc_k"]))

    rpm, torque, proc = np.meshgrid(rpm_grid, torque_grid, proc_grid, indexing="ij")
    rpm, torque, proc = rpm.ravel(), torque.ravel(), proc.ravel()
    # текущий режим тоже кандидат: оптимизатор не может выдать хуже, чем есть
    rpm = np.append(rpm, data["rpm"])
    torque = np.append(torque, data["torque"])
    proc = np.append(proc, data["proc_k"])

    frame = pd.DataFrame({
        "Type": encode_type(data["type_"]),
        "Air temperature [K]": air,
        "Process temperature [K]": proc,
        "Rotational speed [rpm]": rpm,
        "Torque [Nm]": torque,
        "Tool wear [min]": data["wear"],
    })
    X = add_engineered_features(frame)[FEATURES]
    p = _binary_model().predict_proba(X)[:, 1]            # весь перебор одним вызовом

    dev = (np.abs(rpm - data["rpm"]) / 1000
           + np.abs(torque - data["torque"]) / 50
           + np.abs(proc - data["proc_k"]) / 10)

    ok = p <= target
    if ok.any():                                           # самое мягкое изменение, достигающее цели
        idx = int(np.where(ok)[0][np.argmin(dev[ok])])
    else:                                                  # цель недостижима - минимум риска
        idx = int(np.argmin(p + 0.05 * dev))

    return {
        "rpm": float(rpm[idx]),
        "torque": float(torque[idx]),
        "proc_k": float(proc[idx]),
        "risk": float(p[idx]),
    }


# ==========================================
# ЕДИНАЯ ФУНКЦИЯ ДЛЯ FRONTEND
# ==========================================
def analyze_machine(machine: dict, target_max_risk: float = 0.15) -> dict:
    """Вход — данные станка (допустимы ключи 'type' или 'type_'), выход — JSON-совместимый dict.

    machine = {"type_": "L", "air_k": 298.5, "proc_k": 308.7,
               "rpm": 1380, "torque": 62.0, "wear": 210}

    Ответ (ключи фиксированы, все значения - обычные float/str/list/None):
        status           "OPTIMAL"          - риск ниже цели, ничего делать не нужно
                         "ACTION_REQUIRED"  - есть уставки, достигающие цели
                         "CRITICAL"         - цель недостижима уставками (нужна замена/остановка)
        failure_risk     вероятность поломки сейчас, 0..1
        risk_level       "low" | "medium" | "high"
        causes           [{"code": "OSF", "label": "..."}]
        recommended      {"rpm", "torque", "proc_k"} или None
        optimized_risk   риск после применения recommended (или текущий, если рекомендации нет)
        achieved_target  True, если optimized_risk <= target_max_risk
        extra_action     "replace_tool" | None
    """
    if not 0.0 < target_max_risk < 1.0:
        raise MachineValidationError({"target_max_risk": "должно быть между 0 и 1"})
    data = validate_machine(machine)

    failure_risk = float(predict_failure_risk(**data))          # AI1
    _, raw_causes = predict_failure_causes(**data)              # AI2
    codes = [str(c) for c in (raw_causes or [])]
    causes = [{"code": c, "label": CAUSE_LABELS.get(c, c)} for c in codes]

    # износ не лечится ни оборотами, ни моментом
    replace_tool = "TWF" in codes or (data["wear"] >= 200 and failure_risk >= target_max_risk)
    extra_action = "replace_tool" if replace_tool else None

    base = {
        "failure_risk": round(failure_risk, 4),
        "risk_level": _risk_level(failure_risk),
        "causes": causes,
        "extra_action": extra_action,
    }

    if failure_risk < target_max_risk:
        return {**base, "status": "OPTIMAL", "recommended": None,
                "optimized_risk": round(failure_risk, 4), "achieved_target": True}

    best = _find_best_settings(data, target_max_risk)
    improved = best["risk"] < failure_risk - MIN_GAIN
    optimized = best["risk"] if improved else failure_risk
    achieved = optimized <= target_max_risk

    return {
        **base,
        "status": "ACTION_REQUIRED" if achieved else "CRITICAL",
        "recommended": ({"rpm": best["rpm"], "torque": best["torque"], "proc_k": best["proc_k"]}
                        if improved else None),
        "optimized_risk": round(optimized, 4),
        "achieved_target": achieved,
    }


# ==========================================
# ПАКЕТНЫЙ РЕЖИМ (CSV)
# ==========================================
def analyze_batch(machines: list[dict], target_max_risk: float = 0.15,
                  optimize_top: int = 0) -> list[dict]:
    """Риск для множества станков одним вызовом модели. Невалидные строки не роняют пакет:
    для них возвращается {"index", "error"}. optimize_top=N - полный анализ N самых рискованных."""
    rows, results = [], []
    for i, m in enumerate(machines):
        try:
            rows.append((i, validate_machine(m)))
        except MachineValidationError as e:
            results.append({"index": i, "error": e.errors})

    if rows:
        frame = pd.DataFrame({
            "Type": [encode_type(d["type_"]) for _, d in rows],
            "Air temperature [K]": [d["air_k"] for _, d in rows],
            "Process temperature [K]": [d["proc_k"] for _, d in rows],
            "Rotational speed [rpm]": [d["rpm"] for _, d in rows],
            "Torque [Nm]": [d["torque"] for _, d in rows],
            "Tool wear [min]": [d["wear"] for _, d in rows],
        })
        p = _binary_model().predict_proba(add_engineered_features(frame)[FEATURES])[:, 1]
        for (i, _), risk in zip(rows, p):
            results.append({"index": i, "failure_risk": round(float(risk), 4),
                            "risk_level": _risk_level(float(risk))})

    results.sort(key=lambda r: r["index"])
    if optimize_top > 0:
        risky = sorted((r for r in results if "failure_risk" in r),
                       key=lambda r: -r["failure_risk"])[:optimize_top]
        for r in risky:
            r["analysis"] = analyze_machine(machines[r["index"]], target_max_risk)
    return results


# ==========================================
# HTTP API ДЛЯ САЙТА (uvicorn failurdesicion:app --reload --port 8000)
# ==========================================
app = FastAPI(title="Predictive Maintenance API")
ORIGINS = os.getenv(
    "CORS_ORIGINS", "http://localhost:5500,http://127.0.0.1:5500,http://localhost:3000"
).split(",")
app.add_middleware(CORSMiddleware, allow_origins=ORIGINS,
                   allow_methods=["*"], allow_headers=["*"])


class Machine(BaseModel):
    type: Literal["L", "M", "H"]
    air_k: float = Field(ge=290, le=310)
    proc_k: float = Field(ge=300, le=320)
    rpm: float = Field(gt=0, le=4000)
    torque: float = Field(ge=0, le=100)
    wear: float = Field(ge=0, le=300)


@app.exception_handler(MachineValidationError)
def _bad_machine(_, e: MachineValidationError):
    return JSONResponse(status_code=422, content={"errors": e.errors})


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/analyze")
def api_analyze(m: Machine, target: float = Query(0.15, gt=0, lt=1)):
    return analyze_machine(m.model_dump(), target)


@app.post("/batch")
def api_batch(items: List[Machine], target: float = Query(0.15, gt=0, lt=1),
              optimize_top: int = Query(0, ge=0, le=50)):
    return analyze_batch([i.model_dump() for i in items[:5000]], target, optimize_top)


# ==========================================
# ДЕМОНСТРАЦИЯ
# ==========================================
if __name__ == "__main__":
    sensors = {"type_": "L", "air_k": 298.5, "proc_k": 308.7,
               "rpm": 1380, "torque": 62.0, "wear": 210}

    print("=" * 60)
    print("ВХОДЯЩИЕ ДАННЫЕ С СЕНСОРОВ ОБОРУДОВАНИЯ:")
    for k, v in sensors.items():
        print(f"  {k}: {v}")
    print("=" * 60)

    result = analyze_machine(sensors)
    print(f"Риск брака P(Failure): {result['failure_risk'] * 100:.1f}% ({result['risk_level']})")
    print("Причины:", ", ".join(c["label"] for c in result["causes"]) or "не определены")

    rec = result["recommended"]
    if rec:
        print("\nРЕКОМЕНДУЕМЫЕ УСТАВКИ:")
        print(f"   RPM:    {sensors['rpm']} -> {rec['rpm']:.0f}")
        print(f"   Torque: {sensors['torque']} -> {rec['torque']:.1f}")
        print(f"   Temp:   {sensors['proc_k']} -> {rec['proc_k']:.1f}")
        print(f"\nРиск после оптимизации: {result['optimized_risk'] * 100:.1f}%")
    if result["extra_action"] == "replace_tool":
        print("\nТребуется замена инструмента: уставками износ не исправить.")
    if result["status"] == "CRITICAL":
        print("Цель по риску недостижима изменением режима.")
    if result["status"] == "OPTIMAL":
        print("\nПроцесс в норме. Корректировка не требуется.")
    print("=" * 60)
