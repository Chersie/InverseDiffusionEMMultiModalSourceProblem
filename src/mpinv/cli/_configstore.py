"""ConfigStore registration hook.

Currently a no-op. The framework constructs every leaf via ``_target_`` directives in
the YAML files (and validates kwargs at the dataclass constructor), so we do not
register dataclass schemas with Hydra's ``ConfigStore``. ``Literal`` typed fields
are also not supported by OmegaConf's structured-config validator, so registration
would force us to weaken the dataclass signatures with plain ``str`` annotations.

If a future component needs structured-config validation, register it here against
schemas that use plain ``str`` (not ``Literal``).
"""

from __future__ import annotations


def register_configs() -> None:
    return None
