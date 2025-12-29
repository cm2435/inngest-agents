"""FastAPI app with Inngest serve endpoint for testing agents.

Run with:
    uv run uvicorn tests.app:app --reload --port 8000

Then start Inngest dev server (pointing to the app):
    npx inngest-cli@latest dev --port 9001 -u http://localhost:8000/api/inngest

Test events:
    - agent/run: Run legacy data_analysis_agent
    - agent/triage: Run triage_agent (hands off to sales/support)
    - agent/sales: Run sales_agent directly (Pydantic model returns)
    - agent/support: Run support_agent directly
"""

import inngest
import inngest.fast_api
from agents import Runner
from fastapi import FastAPI
from inngest_agents import finalize_run, set_step

from .agent import data_analysis_agent, sales_agent, support_agent, triage_agent

# Dev server configuration
DEV_SERVER_URL = "http://127.0.0.1:9001"

# Create Inngest client - explicitly configured for local dev server
# PydanticSerializer enables proper serialization of Pydantic models in step outputs
inngest_client = inngest.Inngest(
    app_id="inngest-agents-test",
    api_base_url=DEV_SERVER_URL,
    event_api_base_url=DEV_SERVER_URL,
    is_production=False,
    signing_key=None,
    serializer=inngest.PydanticSerializer(),
)


@inngest_client.create_function(
    fn_id="run-agent",
    trigger=inngest.TriggerEvent(event="agent/run"),
)
async def run_agent(ctx: inngest.Context) -> dict:
    """Run the legacy Data Analysis Agent."""
    set_step(ctx.step)
    prompt = str(ctx.event.data.get("prompt", "Analyze Q4 sales data"))

    result = await Runner.run(starting_agent=data_analysis_agent, input=prompt)
    finalized = await finalize_run(result, data_analysis_agent.name)

    return finalized.to_dict()


@inngest_client.create_function(
    fn_id="run-triage",
    trigger=inngest.TriggerEvent(event="agent/triage"),
)
async def run_triage(ctx: inngest.Context) -> dict:
    """Run the Triage Agent (demonstrates handoffs to sub-agents)."""
    set_step(ctx.step)
    prompt = str(ctx.event.data.get("prompt", "I need help with my account"))

    result = await Runner.run(starting_agent=triage_agent, input=prompt)
    finalized = await finalize_run(result, triage_agent.name)

    return finalized.to_dict()


@inngest_client.create_function(
    fn_id="run-sales",
    trigger=inngest.TriggerEvent(event="agent/sales"),
)
async def run_sales(ctx: inngest.Context) -> dict:
    """Run the Sales Agent (demonstrates Pydantic model returns)."""
    set_step(ctx.step)
    prompt = str(ctx.event.data.get("prompt", "Analyze sales for the north region"))

    result = await Runner.run(starting_agent=sales_agent, input=prompt)
    finalized = await finalize_run(result, sales_agent.name)

    return finalized.to_dict()


@inngest_client.create_function(
    fn_id="run-support",
    trigger=inngest.TriggerEvent(event="agent/support"),
)
async def run_support(ctx: inngest.Context) -> dict:
    """Run the Support Agent."""
    set_step(ctx.step)
    prompt = str(
        ctx.event.data.get("prompt", "Customer ABC123 can't access their account, this is urgent")
    )

    result = await Runner.run(starting_agent=support_agent, input=prompt)
    finalized = await finalize_run(result, support_agent.name)

    return finalized.to_dict()


# Create FastAPI app
app = FastAPI(
    title="Inngest Agents Test",
    description="Test app for Inngest-backed OpenAI Agents SDK",
)


# Mount Inngest serve endpoint
inngest.fast_api.serve(
    app,
    inngest_client,
    [run_agent, run_triage, run_sales, run_support],
)


@app.get("/")
async def root():
    """Health check endpoint."""
    return {"status": "ok", "message": "Inngest Agents Test App"}


@app.get("/health")
async def health():
    """Health check for Inngest."""
    return {"status": "healthy"}
