"""Ambient Inngest context for durable tool execution.

This module provides a simple way to access Inngest step tools from within
agent tool functions without needing to pass the step explicitly through
the call chain.

Usage:
    # In your Inngest function handler:
    async def run_agent(ctx: inngest.Context):
        set_step(ctx.step)  # Make step available to tools
        result = await Runner.run(agent, prompt)
        return result

    # Wrap tools at agent creation time:
    @function_tool
    async def fetch_data(source: str) -> str:
        return await api.get(source)

    agent = Agent(
        name="my_agent",
        tools=[as_step(fetch_data), as_step(analyze)],
    )
"""

import json
from collections.abc import Awaitable, Callable
from contextvars import ContextVar
from dataclasses import replace
from typing import TYPE_CHECKING, Any

from agents import FunctionTool
from inngest import NonRetriableError
from pydantic import BaseModel

if TYPE_CHECKING:
    from inngest import Step


_inngest_step: ContextVar["Step | None"] = ContextVar("inngest_step", default=None)


def get_step() -> "Step | None":
    """Get the Inngest step from the current async context.

    Returns:
        The Inngest Step object if running in an Inngest function context,
        None otherwise.
    """
    return _inngest_step.get()


def set_step(step: "Step") -> None:
    """Set the Inngest step for the current async context.

    Call this at the start of your Inngest function handler to make
    the step tools available to all tool functions.

    Args:
        step: The Inngest Step object from the function context.
    """
    _inngest_step.set(step)


async def durable[T](step_id: str, fn: Callable[[], Awaitable[T]]) -> T:
    """Run a function durably if in Inngest context, otherwise run directly.

    This is a convenience wrapper that automatically handles the check for
    whether we're running in an Inngest context. If we are, the function
    is wrapped in step.run() for durability and memoization. If not, the
    function is called directly.

    Args:
        step_id: A unique identifier for this step. Should be deterministic
            for proper memoization on retries.
        fn: An async function to execute. Should be a lambda or function
            that takes no arguments.

    Returns:
        The result of calling fn().

    Example:
        async def fetch_weather(city: str) -> dict:
            # API call here
            return {"temp": 72}

        result = await durable(f"weather_{city}", lambda: fetch_weather(city))
    """
    step = get_step()
    if step:
        result = await step.run(step_id, fn)
        # step.run returns T | None, but None only occurs in edge cases
        # We trust that if fn returns T, step.run will also return T
        return result  # type: ignore[return-value]
    return await fn()


def _serialize_for_inngest(value: Any) -> Any:
    """Serialize a value for Inngest storage, handling Pydantic models."""
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, str):
        # Try to parse JSON strings for cleaner display
        try:
            return json.loads(value)
        except (json.JSONDecodeError, TypeError):
            pass
    return value


def as_step(tool: FunctionTool) -> FunctionTool:
    """Wrap a FunctionTool to run as an Inngest step.

    The wrapped tool will:
    - Run as a durable step if in Inngest context (memoized on retry)
    - Run directly if not in Inngest context
    - Raise NonRetriableError on any exception (prevents infinite retries)

    Args:
        tool: A FunctionTool (from @function_tool decorator).

    Returns:
        A new FunctionTool with the same signature but durable execution.

    Example:
        @function_tool
        async def fetch_data(source: str) -> str:
            return await api.get(source)

        @function_tool
        def analyze(data: str) -> str:
            return f"Analysis: {data}"

        # Wrap at agent creation time
        agent = Agent(
            tools=[as_step(fetch_data), as_step(analyze)],
        )
    """
    original_invoke = tool.on_invoke_tool

    async def wrapped_invoke(ctx: Any, args: str) -> Any:
        step = get_step()

        async def _execute() -> dict[str, Any]:
            """Execute tool and return structured result with input/output."""
            try:
                result = await original_invoke(ctx, args)
                return {
                    "input": _serialize_for_inngest(args),
                    "output": _serialize_for_inngest(result),
                }

            except NonRetriableError:
                raise  # Don't double-wrap

            except Exception as e:
                raise NonRetriableError(f"{tool.name} failed: {e}") from e

        if step:
            step_id = f"tool_{tool.name}"
            step_result = await step.run(step_id, _execute)
            # Return just the output to the agent, but Inngest stores full dict
            return step_result["output"] if step_result else None

        # Outside Inngest context, just run directly
        direct_result = await _execute()
        return direct_result["output"]

    return replace(tool, on_invoke_tool=wrapped_invoke)
