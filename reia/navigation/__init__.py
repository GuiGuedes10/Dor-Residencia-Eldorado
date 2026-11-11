"""
REIA Web Navigation Subsystem.

Provides automated browser navigation, deterministic offline mock execution,
and visual/textual artifact persistence for PUC-Campinas Vestibular portal routes.
"""

from __future__ import annotations

from reia.navigation.artifacts import ArtifactEngine, generate_png_bytes
from reia.navigation.base import BaseNavigator, NavigationResult
from reia.navigation.mock_driver import MockNavigator
from reia.navigation.routes import (
    CANONICAL_ROUTES,
    get_navigator,
    get_target_url,
    is_valid_intent,
    resolve_intent_from_url,
)
from reia.navigation.selenium_driver import (
    LocalSeleniumNavigator,
    SeleniumNavigator,
    build_chrome_options,
)

__all__ = [
    "ArtifactEngine",
    "BaseNavigator",
    "CANONICAL_ROUTES",
    "LocalSeleniumNavigator",
    "MockNavigator",
    "NavigationResult",
    "SeleniumNavigator",
    "build_chrome_options",
    "generate_png_bytes",
    "get_navigator",
    "get_target_url",
    "is_valid_intent",
    "resolve_intent_from_url",
]
