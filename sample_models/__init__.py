"""
Namespace for optional custom PII detection models.

Each module must define a callable `detect_pii(text: str, language: str = "en")`
returning a list of entity dictionaries. Files starting with `_` are ignored.
See README.md in this directory for implementation details.
"""


