"""Data models for Inngest agent integration."""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RunStats:
    """Aggregate statistics from an agent run."""

    total_tool_calls: int
    total_messages: int
    total_reasoning_steps: int
    total_items: int
    agents_involved: list[str]
    num_agents: int
    starting_agent: str
    final_agent: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    total_cost_usd: float | None
    model: str

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict for JSON serialization."""
        return {k: v for k, v in self.__dict__.items()}


@dataclass(frozen=True)
class FinalizedRun:
    """Result of a finalized agent run with stats."""

    final_output: Any
    stats: RunStats

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict for JSON serialization."""
        return {
            "final_output": self.final_output,
            "stats": self.stats.to_dict(),
        }
