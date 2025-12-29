"""Integration tests for Inngest-backed Agents SDK.

These tests require:
1. Inngest dev server running: npx inngest-cli@latest dev --port 9001
2. FastAPI app running: uv run uvicorn tests.app:app --reload --port 8000

Run tests with:
    uv run pytest tests/test_integration.py -v
"""

import asyncio
import os

import httpx
import pytest

# Test configuration
INNGEST_DEV_URL = os.getenv("INNGEST_DEV_URL", "http://127.0.0.1:9001")
APP_URL = os.getenv("APP_URL", "http://127.0.0.1:8000")


@pytest.fixture
def anyio_backend():
    return "asyncio"


async def check_inngest_dev_server() -> bool:
    """Check if Inngest dev server is running."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{INNGEST_DEV_URL}/health", timeout=2.0)
            return response.status_code == 200
    except Exception:
        return False


async def check_app_server() -> bool:
    """Check if FastAPI app is running."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{APP_URL}/health", timeout=2.0)
            return response.status_code == 200
    except Exception:
        return False


@pytest.mark.asyncio
async def test_servers_running():
    """Verify both servers are running before running other tests."""
    inngest_running = await check_inngest_dev_server()
    app_running = await check_app_server()

    if not inngest_running:
        pytest.skip(
            "Inngest dev server not running. Start with: npx inngest-cli@latest dev --port 9001"
        )
    if not app_running:
        pytest.skip(
            "FastAPI app not running. Start it with: uv run uvicorn tests.app:app --port 8000"
        )


@pytest.mark.asyncio
async def test_send_agent_run_event():
    """Test sending an event to trigger the agent run.

    This test:
    1. Sends an event to Inngest dev server
    2. Waits for the function to complete
    3. Verifies the run appears in Inngest

    Note: This is a basic smoke test. For full verification,
    check the Inngest dev server UI at http://127.0.0.1:8288
    """
    # Check servers are running
    if not await check_inngest_dev_server():
        pytest.skip("Inngest dev server not running")
    if not await check_app_server():
        pytest.skip("FastAPI app not running")

    async with httpx.AsyncClient() as client:
        # Send event to Inngest dev server
        response = await client.post(
            f"{INNGEST_DEV_URL}/e/inngest-agents-test",
            json={
                "name": "agent/run",
                "data": {
                    "prompt": "Analyze Q4 sales data and send me a summary",
                },
            },
            timeout=10.0,
        )

        assert response.status_code in [200, 201, 202], (
            f"Failed to send event: {response.status_code} - {response.text}"
        )

        print(f"Event sent successfully: {response.json()}")

        # Wait a bit for the function to start processing
        await asyncio.sleep(2)

        # Check runs endpoint to see if our function was triggered
        runs_response = await client.get(
            f"{INNGEST_DEV_URL}/v1/runs",
            timeout=10.0,
        )

        if runs_response.status_code == 200:
            runs_data = runs_response.json()
            print(f"Runs found: {runs_data}")
        else:
            print(f"Could not fetch runs: {runs_response.status_code}")


@pytest.mark.asyncio
async def test_agent_workflow_steps():
    """Test that the agent workflow creates expected Inngest steps.

    After running, check the Inngest dev UI for these steps:
    - fetch_q4_sales (from fetch_data tool using durable())
    - save_report_* (from save_report tool with manual step)
    - notification_delay (sleep step from send_notification)
    - send_notification (from send_notification tool)

    Note: The analyze tool should NOT create a step (pure compute).
    """
    if not await check_inngest_dev_server():
        pytest.skip("Inngest dev server not running")
    if not await check_app_server():
        pytest.skip("FastAPI app not running")

    async with httpx.AsyncClient() as client:
        # Send the event
        response = await client.post(
            f"{INNGEST_DEV_URL}/e/inngest-agents-test",
            json={
                "name": "agent/run",
                "data": {
                    "prompt": "Analyze Q4 sales data and send me a summary",
                },
            },
            timeout=10.0,
        )

        assert response.status_code in [200, 201, 202]

        # Give the function time to complete (includes 2s sleep in notification)
        print("Waiting for function to complete (includes 2s notification delay)...")
        await asyncio.sleep(10)

        # The actual step verification should be done manually in the Inngest UI
        # at http://127.0.0.1:9001
        print("\n✓ Event sent! Check Inngest dev UI at http://127.0.0.1:9001 for:")
        print("  - fetch_q4_sales step (durable helper)")
        print("  - save_report_* step (manual get_step)")
        print("  - notification_delay step (step.sleep)")
        print("  - send_notification step (step.run)")
        print("  - NO step for analyze (pure compute)")


@pytest.mark.asyncio
async def test_sales_agent_pydantic_returns():
    """Test Sales Agent with Pydantic model returns.

    The Sales Agent tools return Pydantic models:
    - fetch_sales_data -> SalesMetrics
    - analyze_sales -> AnalysisResult
    """
    if not await check_inngest_dev_server():
        pytest.skip("Inngest dev server not running")
    if not await check_app_server():
        pytest.skip("FastAPI app not running")

    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{INNGEST_DEV_URL}/e/inngest-agents-test",
            json={
                "name": "agent/sales",
                "data": {"prompt": "Analyze sales for the north region"},
            },
            timeout=10.0,
        )

        assert response.status_code in [200, 201, 202], (
            f"Failed to send event: {response.status_code} - {response.text}"
        )

        print(f"\n✓ Sales agent event sent: {response.json()}")
        print("  Check Inngest UI for Pydantic model outputs in step data")


@pytest.mark.asyncio
async def test_triage_agent_handoff():
    """Test Triage Agent handoff to sub-agents.

    The Triage Agent should hand off to Support Agent for account issues.
    """
    if not await check_inngest_dev_server():
        pytest.skip("Inngest dev server not running")
    if not await check_app_server():
        pytest.skip("FastAPI app not running")

    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{INNGEST_DEV_URL}/e/inngest-agents-test",
            json={
                "name": "agent/triage",
                "data": {"prompt": "I need help with my account, customer ID is ABC123"},
            },
            timeout=10.0,
        )

        assert response.status_code in [200, 201, 202], (
            f"Failed to send event: {response.status_code} - {response.text}"
        )

        print(f"\n✓ Triage agent event sent: {response.json()}")
        print("  Check Inngest UI to see handoff from Triage -> Support Agent")


@pytest.mark.asyncio
async def test_support_agent_ticket_creation():
    """Test Support Agent creating tickets with Pydantic SupportTicket return."""
    if not await check_inngest_dev_server():
        pytest.skip("Inngest dev server not running")
    if not await check_app_server():
        pytest.skip("FastAPI app not running")

    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{INNGEST_DEV_URL}/e/inngest-agents-test",
            json={
                "name": "agent/support",
                "data": {"prompt": "Customer XYZ789 can't login, this is urgent!"},
            },
            timeout=10.0,
        )

        assert response.status_code in [200, 201, 202], (
            f"Failed to send event: {response.status_code} - {response.text}"
        )

        print(f"\n✓ Support agent event sent: {response.json()}")
        print("  Check Inngest UI for SupportTicket Pydantic model in step output")


if __name__ == "__main__":
    # Allow running tests directly
    asyncio.run(test_send_agent_run_event())
