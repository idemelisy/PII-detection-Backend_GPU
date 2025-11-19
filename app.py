#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import importlib.util
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from flask import Flask, jsonify, request
from flask_cors import CORS
from presidio_analyzer import AnalyzerEngine

MODEL_FN = Callable[[str, str], List[Dict[str, Any]]]

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODELS_DIR = BASE_DIR / "sample_models"
ENV_MODELS_DIR = os.environ.get("PII_SAMPLE_MODELS_DIR")
DEFAULT_MODEL_KEY = os.environ.get("PII_DEFAULT_MODEL", "presidio").lower()

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return False


def _coerce_dict(payload: Any) -> Dict[str, Any]:
    return payload if isinstance(payload, dict) else {}


@dataclass
class ModelRecord:
    key: str
    name: str
    detect: MODEL_FN
    metadata: Dict[str, Any] = field(default_factory=dict)
    source: str = "unknown"
    builtin: bool = False


class ModelRegistry:
    def __init__(self, default_key: Optional[str] = None):
        self._models: Dict[str, ModelRecord] = {}
        self.errors: List[Dict[str, str]] = []
        self.default_key = default_key

    def register(
        self,
        *,
        key: str,
        name: str,
        detect_fn: MODEL_FN,
        metadata: Optional[Dict[str, Any]] = None,
        source: str = "",
        builtin: bool = False,
        allow_replace: bool = False,
    ) -> bool:
        normalized_key = key.lower()
        if normalized_key in self._models and not allow_replace:
            logger.warning(
                "Model key '%s' already registered (source=%s). Skipping new source=%s",
                normalized_key,
                self._models[normalized_key].source,
                source,
            )
            return False

        self._models[normalized_key] = ModelRecord(
            key=normalized_key,
            name=name,
            detect=detect_fn,
            metadata=_coerce_dict(metadata),
            source=source,
            builtin=builtin,
        )

        if not self.default_key:
            self.default_key = normalized_key

        logger.info("Registered model '%s' from %s", normalized_key, source or "unknown")
        return True

    def get(self, key: Optional[str]) -> Optional[ModelRecord]:
        if not key:
            return None
        return self._models.get(key.lower())

    def list_models(self) -> List[Dict[str, Any]]:
        entries = []
        for record in sorted(self._models.values(), key=lambda r: r.key):
            entries.append(
                {
                    "key": record.key,
                    "name": record.name,
                    "metadata": record.metadata,
                    "source": record.source,
                    "builtin": record.builtin,
                    "is_default": record.key == self.default_key,
                }
            )
        return entries

    def available_keys(self) -> List[str]:
        return sorted(self._models.keys())


registry = ModelRegistry(default_key=DEFAULT_MODEL_KEY or None)


def _safe_round(value: Any, digits: int = 4) -> Optional[float]:
    if value is None:
        return None
    try:
        return round(float(value), digits)
    except (TypeError, ValueError):
        return None


def _normalize_entities(raw_entities: Any, source_text: str) -> List[Dict[str, Any]]:
    if not isinstance(raw_entities, list):
        return []

    normalized: List[Dict[str, Any]] = []
    for item in raw_entities:
        if not isinstance(item, dict):
            continue

        entity_type = item.get("type") or item.get("label")
        start = item.get("start")
        end = item.get("end")

        if entity_type is None or start is None or end is None:
            continue

        try:
            start_idx = int(start)
            end_idx = int(end)
        except (TypeError, ValueError):
            continue

        value = (
            item.get("text")
            or item.get("value")
            or source_text[start_idx:end_idx]
        )

        normalized.append(
            {
                "type": entity_type,
                "start": start_idx,
                "end": end_idx,
                "text": value,
                "score": _safe_round(item.get("score") or item.get("confidence")),
            }
        )

    return normalized


# ---- Built-in Presidio runner ------------------------------------------------
try:
    analyzer = AnalyzerEngine()
    logger.info("Presidio analyzer initialized")
except Exception as exc:  # pragma: no cover - logging side effect
    analyzer = None
    logger.error("Failed to initialize Presidio analyzer: %s", exc)


def presidio_detect(text: str, language: str = "en") -> List[Dict[str, Any]]:
    if analyzer is None:
        raise RuntimeError("Presidio analyzer is not available on this host")

    if not isinstance(text, str) or not text.strip():
        return []

    results = analyzer.analyze(
        text=text,
        language=language,
        entities=[
            "PHONE_NUMBER",
            "EMAIL_ADDRESS",
            "PERSON",
            "LOCATION",
            "CREDIT_CARD",
            "IBAN_CODE",
            "TR_IDENTITY_NUMBER",
        ],
    )

    return [
        {
            "type": r.entity_type,
            "start": r.start,
            "end": r.end,
            "text": text[r.start : r.end],
            "score": r.score,
        }
        for r in results
    ]


if analyzer is not None:
    registry.register(
        key="presidio",
        name="Presidio",
        detect_fn=presidio_detect,
        metadata={"provider": "Microsoft Presidio"},
        source="builtin",
        builtin=True,
        allow_replace=True,  # allows ALLOW_DEFAULT_OVERRIDE modules to take over
    )


# ---- Dynamic model discovery -------------------------------------------------
def _iter_model_paths() -> List[Path]:
    paths: List[Path] = []

    if ENV_MODELS_DIR:
        for chunk in ENV_MODELS_DIR.split(os.pathsep):
            expanded = Path(chunk).expanduser().resolve()
            if expanded not in paths:
                paths.append(expanded)

    default_dir = DEFAULT_MODELS_DIR.resolve()
    if default_dir not in paths:
        paths.append(default_dir)

    return paths


def _import_module(module_path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if not spec or not spec.loader:
        raise ImportError(f"Cannot load spec for {module_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


def _load_model_from_module(module, file_path: Path) -> None:
    detect_fn = getattr(module, "detect_pii", None)
    if not callable(detect_fn):
        raise AttributeError("Module is missing callable detect_pii(text, language)")

    model_name = getattr(module, "MODEL_NAME", file_path.stem)
    allow_override = _coerce_bool(getattr(module, "ALLOW_DEFAULT_OVERRIDE", False))
    metadata = _coerce_dict(getattr(module, "MODEL_METADATA", {}))

    registered = registry.register(
        key=str(model_name),
        name=str(model_name),
        detect_fn=detect_fn,
        metadata=metadata,
        source=str(file_path),
        builtin=False,
        allow_replace=allow_override,
    )

    if not registered:
        raise RuntimeError(
            f"Model key '{model_name}' already exists and override not allowed"
        )


def discover_custom_models() -> None:
    for directory in _iter_model_paths():
        if not directory.exists():
            logger.info("Model directory %s does not exist; skipping", directory)
            continue

        for path in sorted(directory.iterdir()):
            if path.name.startswith("_"):
                continue

            target: Optional[Path] = None
            if path.is_file() and path.suffix == ".py":
                target = path
            elif path.is_dir() and (path / "__init__.py").exists():
                target = path / "__init__.py"

            if not target:
                continue

            module_name = f"pii_model_{path.stem}_{abs(hash(str(path)))}"
            try:
                module = _import_module(target, module_name)
                _load_model_from_module(module, target)
            except Exception as exc:
                error_msg = f"{target}: {exc}"
                registry.errors.append({"source": str(target), "error": str(exc)})
                logger.error("Failed to load model from %s: %s", target, exc)


discover_custom_models()

if registry.default_key not in registry._models and registry._models:
    registry.default_key = next(iter(registry._models))


# ---- Flask routes ------------------------------------------------------------
@app.route("/")
def index():
    return jsonify({"service": "PII Detection API", "status": "running"})


@app.route("/health")
def health():
    overall_status = "healthy" if registry._models else "no-models"
    return jsonify(
        {
            "status": overall_status,
            "default_model": registry.default_key,
            "models": registry.list_models(),
            "load_errors": registry.errors,
        }
    )


@app.route("/detect-pii", methods=["POST"])
def detect_pii_route():
    payload = request.get_json(silent=True) or {}
    text = payload.get("text", "")
    language = payload.get("language", "en")
    requested_model = payload.get("model") or request.args.get("model")

    if not isinstance(text, str):
        return jsonify({"error": "Field 'text' must be a string"}), 400

    model = registry.get(requested_model) or registry.get(registry.default_key)
    if not model:
        return (
            jsonify(
                {
                    "error": "No PII detection models are available",
                    "available_models": registry.available_keys(),
                }
            ),
            500,
        )

    if not text.strip():
        return jsonify(
            {
                "model_key": model.key,
                "model_used": model.name,
                "total_entities": 0,
                "detected_entities": [],
                "available_models": registry.available_keys(),
            }
        )

    try:
        raw_entities = model.detect(text, language)
    except Exception as exc:
        logger.exception("Model '%s' failed during detection", model.name)
        return (
            jsonify(
                {
                    "error": f"Model '{model.name}' failed: {exc}",
                    "model_key": model.key,
                }
            ),
            500,
        )

    entities = _normalize_entities(raw_entities, text)
    return jsonify(
        {
            "model_key": model.key,
            "model_used": model.name,
            "model_source": model.source,
            "total_entities": len(entities),
            "detected_entities": entities,
            "available_models": registry.available_keys(),
        }
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000)
