"""Escalate a Pydantic AI agent to a stronger model when output validation keeps failing."""

from ._capability import EscalateOnOutputRetry, EscalationBudgetWarning, Level, LevelSpec

__all__ = ('EscalateOnOutputRetry', 'EscalationBudgetWarning', 'Level', 'LevelSpec')
