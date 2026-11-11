"""
Base interface contracts and data models for REIA Web Navigation Subsystem.

Provides:
- NavigationResult: Strongly-typed dataclass containing navigation outcomes,
  page titles, extracted DOM headings, screenshot and metadata artifact paths.
- BaseNavigator: Abstract base class defining the web navigation contract
  supporting both direct URL navigation and semantic intent routing.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class NavigationResult:
    """Strongly-typed output contract for web navigation automation.

    Matches interface specification from PROJECT.md and pipeline integration contracts.
    """
    success: bool
    intent_name: str
    target_url: str
    page_title: str
    screenshot_path: str
    metadata_path: str
    headings: List[str]
    links: List[Dict[str, str]] = field(default_factory=list)
    error_message: Optional[str] = None
    load_time_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert navigation result to a serializable dictionary."""
        return {
            "success": self.success,
            "intent_name": self.intent_name,
            "target_url": self.target_url,
            "page_title": self.page_title,
            "screenshot_path": self.screenshot_path,
            "metadata_path": self.metadata_path,
            "headings": list(self.headings),
            "links": list(self.links),
            "error_message": self.error_message,
            "load_time_seconds": self.load_time_seconds,
        }


class BaseNavigator(ABC):
    """Abstract Base Class for web navigation drivers in REIA.

    Implementations include:
    - SeleniumNavigator: Live headless Chrome automation using standard Selenium 4.
    - MockNavigator: Deterministic offline navigator using local HTML fixtures and
      zero-dependency pure-Python PNG generation.
    """

    @abstractmethod
    def navigate(self, target_url: str, intent_name: str) -> NavigationResult:
        """Navigate to a target URL associated with an identified user intent.

        Args:
            target_url: Destination URL to load in the browser.
            intent_name: Canonical intent identifier (e.g. 'GUIA_UNIVERSITARIO').

        Returns:
            NavigationResult instance with execution status and captured artifacts.
        """
        pass

    def navigate_intent(self, intent_name: str) -> NavigationResult:
        """Navigate based directly on a canonical intent identifier.

        Resolves the target URL from configuration and delegates to navigate().

        Args:
            intent_name: Canonical intent identifier.

        Returns:
            NavigationResult instance.
        """
        from reia.navigation.routes import get_target_url

        target_url = get_target_url(intent_name)
        return self.navigate(target_url=target_url, intent_name=intent_name)

    def __call__(self, intent_name: str) -> NavigationResult:
        """Callable convenience shortcut delegating to navigate_intent."""
        return self.navigate_intent(intent_name)
