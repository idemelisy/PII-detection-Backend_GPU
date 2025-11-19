# Sample Model Integrations

Place any GPU-backed or custom NLP models in this directory. Each model runs
inside the Flask backend and must expose a simple, synchronous interface so
that `/detect-pii` can call it just like the built-in Presidio analyzer.

## Folder Layout

```
sample_models/
├── README.md          # You are here
├── piranha.py         # Example – must define detect_pii()
├── nemo/              # Optional package form (needs __init__.py)
│   └── __init__.py    # detect_pii implementation
└── ...
```

## Required Interface

Every model file (or package) must define a callable named `detect_pii`:

```python
def detect_pii(text: str, language: str = "en") -> list[dict]:
    ...
```

Return a list of dictionaries using the schema below. Extra fields are ignored.

| Field       | Type    | Description                                  |
|-------------|---------|----------------------------------------------|
| `type`      | str     | Entity type (e.g. `PERSON`, `EMAIL`)         |
| `start`     | int     | Start index inside the input text            |
| `end`       | int     | End index (exclusive)                        |
| `value`     | str     | Text span (optional, backend can infer it)   |
| `confidence`| float   | Confidence score between 0 and 1 (optional)  |

You can optionally expose:

- `MODEL_NAME = "friendly-name"` – overrides the filename-derived key.
- `MODEL_METADATA = {"provider": "...", "device": "...", ...}` – displayed in the health response.
- `ALLOW_DEFAULT_OVERRIDE = True` – only needed if `MODEL_NAME` equals the builtin
  default (`presidio`). When enabled, this module replaces the shipped Presidio
  runner but the backend will fall back to the builtin implementation if yours
  fails.

## Loading Rules

* Files beginning with `_` are ignored.
* Directories without an `__init__.py` are skipped.
* Failures are logged and surfaced through `/health`.
* Missing custom models never block the backend – it falls back to Presidio.

## Example Skeleton

```python
# sample_models/piranha.py
MODEL_NAME = "piranha"
MODEL_METADATA = {"provider": "Local GPU", "device": "A100"}

def detect_pii(text: str, language: str = "en") -> list[dict]:
    # Load or reuse your model here
    predictions = run_model(text)
    return [
        {
            "type": pred.label,
            "start": pred.start,
            "end": pred.end,
            "value": pred.text,
            "confidence": pred.score,
        }
        for pred in predictions
    ]
```

Copy your sample scripts into this folder (or update `PII_SAMPLE_MODELS_DIR`)
and restart `app.py`. The backend will detect them automatically, and the
Chrome extension can request them via the `model` field.


