"""Run statistics and finalization for agent observability."""

from typing import Any

from agents.items import MessageOutputItem, ReasoningItem, ToolCallItem
from agents.result import RunResult
from pydantic import BaseModel

from .context import get_step
from .schemas import FinalizedRun, RunStats

try:
    from litellm.cost_calculator import cost_per_token

    CAN_CALCULATE_COST = True
except ImportError:
    CAN_CALCULATE_COST = False


def _serialize_for_inngest(value: Any) -> Any:
    """Serialize a value for Inngest storage, handling Pydantic models."""
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, str):
        # Try to parse JSON strings for cleaner display
        import json

        try:
            return json.loads(value)
        except (json.JSONDecodeError, TypeError):
            pass
    return value


async def finalize_run(
    result: RunResult,
    starting_agent_name: str,
    model_name: str = "gpt-4o",
) -> FinalizedRun:
    """Finalize an agent run by logging aggregate stats as an Inngest step.

    Call this after Runner.run() completes. It will:
    1. Compute aggregate stats (tool calls, tokens, cost, agents)
    2. Store them in a final 'run_stats' step for observability
    3. Return a FinalizedRun with final_output and stats

    Args:
        result: The RunResult from Runner.run()
        starting_agent_name: Name of the starting agent
        model_name: Model name for cost calculation (default: gpt-4o)

    Returns:
        FinalizedRun with final_output and stats

    Example:
        result = await Runner.run(starting_agent=agent, input=prompt)
        finalized = await finalize_run(result, agent.name)
        return finalized.to_dict()  # For Inngest JSON serialization
    """
    stats = _compute_run_stats(result, starting_agent_name, model_name)

    step = get_step()
    if step:

        async def _return_stats() -> dict[str, Any]:
            return stats.to_dict()

        await step.run("run_stats", _return_stats)

    return FinalizedRun(
        final_output=_serialize_for_inngest(result.final_output),
        stats=stats,
    )


def _compute_run_stats(
    result: RunResult,
    starting_agent_name: str,
    model_name: str,
) -> RunStats:
    """Compute aggregate statistics from a RunResult."""
    usage = result.context_wrapper.usage

    # Count items by type
    tool_calls = 0
    messages = 0
    reasoning_steps = 0

    for item in result.new_items:
        if isinstance(item, ToolCallItem):
            tool_calls += 1
        elif isinstance(item, MessageOutputItem):
            messages += 1
        elif isinstance(item, ReasoningItem):
            reasoning_steps += 1

    # Track agents involved
    agents_involved = {starting_agent_name}
    if result.last_agent and result.last_agent.name != starting_agent_name:
        agents_involved.add(result.last_agent.name)

    # Token counts
    input_tokens = usage.input_tokens
    output_tokens = usage.output_tokens
    total_tokens = input_tokens + output_tokens

    # Cost calculation (uses litellm if available)
    total_cost_usd = _estimate_cost(input_tokens, output_tokens, model_name)

    return RunStats(
        total_tool_calls=tool_calls,
        total_messages=messages,
        total_reasoning_steps=reasoning_steps,
        total_items=len(result.new_items),
        agents_involved=sorted(agents_involved),
        num_agents=len(agents_involved),
        starting_agent=starting_agent_name,
        final_agent=result.last_agent.name if result.last_agent else starting_agent_name,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        total_cost_usd=total_cost_usd,
        model=model_name,
    )


def _estimate_cost(input_tokens: int, output_tokens: int, model_name: str) -> float | None:
    """Estimate cost based on token counts using litellm.

    Returns None if litellm is not available or cost calculation fails.
    """
    if not CAN_CALCULATE_COST:
        return None
    try:
        prompt_cost, completion_cost = cost_per_token(
            model=model_name,
            prompt_tokens=input_tokens,
            completion_tokens=output_tokens,
        )
        return prompt_cost + completion_cost
    except Exception:
        return None
