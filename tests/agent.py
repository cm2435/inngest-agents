"""Test agents demonstrating Inngest integration patterns.

Tests:
1. Pydantic model returns - structured data from tools
2. Agent handoffs - triage agent delegates to specialized sub-agents
3. Multi-step workflows with durability

Test prompts:
- "Analyze Q4 sales data" -> Sales Agent
- "I need help with my account" -> Support Agent
- "What's the refund policy?" -> Triage decides
"""

from enum import Enum

from agents import Agent, function_tool, handoff
from inngest_agents import as_step
from pydantic import BaseModel

# ============================================================================
# Pydantic Models for structured tool returns
# ============================================================================


class SalesMetrics(BaseModel):
    """Structured sales data."""

    q4_sales: int
    q3_sales: int
    growth_rate: float
    top_products: list[str]
    region: str


class AnalysisResult(BaseModel):
    """Structured analysis result."""

    summary: str
    recommendation: str
    confidence: float
    metrics: dict[str, float]


class TicketPriority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"


class SupportTicket(BaseModel):
    """Structured support ticket."""

    ticket_id: str
    customer_id: str
    issue: str
    priority: TicketPriority
    assigned_to: str


# ============================================================================
# Sales Agent - handles sales data analysis
# ============================================================================


@function_tool
async def fetch_sales_data(region: str) -> SalesMetrics:
    """Fetch sales metrics for a region. Returns structured Pydantic model.

    Args:
        region: The sales region (e.g., "north", "south", "west", "east")
    """
    # Simulated API call returning Pydantic model
    return SalesMetrics(
        q4_sales=150000,
        q3_sales=120000,
        growth_rate=0.25,
        top_products=["Widget A", "Widget B", "Widget C"],
        region=region,
    )


@function_tool
def analyze_sales(metrics: SalesMetrics) -> AnalysisResult:
    """Analyze sales metrics and return structured result.

    Args:
        metrics: The sales metrics to analyze
    """
    growth_pct = metrics.growth_rate * 100
    is_growing = growth_pct > 20

    return AnalysisResult(
        summary=f"Q4 sales for {metrics.region}: ${metrics.q4_sales:,} ({growth_pct:.1f}% growth)",
        recommendation="Expand production" if is_growing else "Maintain current levels",
        confidence=0.85 if is_growing else 0.72,
        metrics={
            "q4_sales": float(metrics.q4_sales),
            "q3_sales": float(metrics.q3_sales),
            "growth_rate": metrics.growth_rate,
        },
    )


@function_tool
async def save_analysis(summary: str, recommendation: str, report_name: str) -> str:
    """Save analysis result to storage.

    Args:
        summary: The analysis summary
        recommendation: The recommendation from analysis
        report_name: Name for the report
    """
    return f"Saved '{report_name}': {summary} - {recommendation}"


SALES_AGENT_PROMPT = """\
You are a sales data analyst. When asked to analyze sales:
1. Fetch sales data for the relevant region using fetch_sales_data
2. Analyze the metrics using analyze_sales
3. Save the result using save_analysis

Be thorough and provide actionable insights."""


sales_agent = Agent(
    name="Sales Agent",
    instructions=SALES_AGENT_PROMPT,
    handoff_description="Handles sales data analysis and reporting",
    tools=[
        as_step(fetch_sales_data),
        as_step(analyze_sales),
        as_step(save_analysis),
    ],
)


# ============================================================================
# Support Agent - handles customer support tickets
# ============================================================================


@function_tool
async def create_ticket(
    customer_id: str,
    issue: str,
    priority: TicketPriority,
) -> SupportTicket:
    """Create a support ticket. Returns structured Pydantic model.

    Args:
        customer_id: The customer's ID
        issue: Description of the issue
        priority: Ticket priority level
    """
    import uuid

    return SupportTicket(
        ticket_id=f"TKT-{uuid.uuid4().hex[:8].upper()}",
        customer_id=customer_id,
        issue=issue,
        priority=priority,
        assigned_to="support_team",
    )


@function_tool
async def lookup_customer(customer_id: str) -> dict:
    """Look up customer information.

    Args:
        customer_id: The customer's ID
    """
    return {
        "customer_id": customer_id,
        "name": "John Doe",
        "account_type": "premium",
        "since": "2023-01-15",
    }


@function_tool
async def send_response(ticket_id: str, message: str) -> str:
    """Send a response to the customer.

    Args:
        ticket_id: The ticket ID
        message: Response message to send
    """
    return f"Response sent for {ticket_id}: {message[:50]}..."


SUPPORT_AGENT_PROMPT = """\
You are a customer support specialist. You MUST complete ALL 3 steps in order:

1. FIRST: Look up the customer using lookup_customer with their ID
2. THEN: Create a ticket using create_ticket with appropriate priority
3. FINALLY: Send a response using send_response with the ticket ID

You MUST call all 3 tools. Do not skip any steps."""


support_agent = Agent(
    name="Support Agent",
    instructions=SUPPORT_AGENT_PROMPT,
    handoff_description="Handles customer support issues, account problems, and complaints",
    tools=[
        as_step(create_ticket),
        as_step(lookup_customer),
        as_step(send_response),
    ],
)


# ============================================================================
# Triage Agent - routes to appropriate sub-agent
# ============================================================================


TRIAGE_AGENT_PROMPT = """\
You are a triage agent. Your job is to understand the user's request and hand off \
to the appropriate specialist:

- For sales data, analysis, or reporting questions -> hand off to Sales Agent
- For account issues, complaints, or support needs -> hand off to Support Agent

Ask clarifying questions if needed, but prefer to hand off quickly."""


triage_agent = Agent(
    name="Triage Agent",
    instructions=TRIAGE_AGENT_PROMPT,
    handoffs=[
        handoff(sales_agent),
        handoff(support_agent),
    ],
)


# ============================================================================
# Legacy data_analysis_agent for backwards compatibility
# ============================================================================


@function_tool
async def fetch_data(source: str) -> dict:
    """Fetch data from a source.

    Args:
        source: The data source to fetch from
    """
    return {
        "q4_sales": 150000,
        "q3_sales": 120000,
        "growth": 0.25,
        "top_products": ["Widget A", "Widget B", "Widget C"],
    }


@function_tool
def analyze(
    q4_sales: int,
    q3_sales: int,
    growth: float,
    top_products: list[str],
) -> str:
    """Analyze sales data.

    Args:
        q4_sales: Q4 sales amount
        q3_sales: Q3 sales amount
        growth: Growth rate as decimal
        top_products: List of top product names
    """
    growth_pct = growth * 100
    return f"""Analysis Results:
- Q4 Sales: ${q4_sales:,}
- Q3 Sales: ${q3_sales:,}  
- Growth: {growth_pct:.1f}%
- Top Products: {", ".join(top_products)}
- Recommendation: {"Expand production" if growth_pct > 20 else "Maintain current levels"}"""


@function_tool
async def save_report(report_name: str, content: str) -> str:
    """Save a report.

    Args:
        report_name: Name for the report
        content: The report content
    """
    return f"Report '{report_name}' saved ({len(content)} chars)"


@function_tool
async def send_notification(message: str) -> str:
    """Send a notification.

    Args:
        message: The notification message
    """
    return f"Notification sent: {message}"


data_analysis_agent = Agent(
    name="Data Analysis Agent",
    instructions="""\
You are a data analyst. Complete these steps:
1. Fetch data using fetch_data
2. Analyze using analyze
3. Save using save_report
4. Notify using send_notification""",
    tools=[
        as_step(fetch_data),
        as_step(analyze),
        as_step(save_report),
        as_step(send_notification),
    ],
)
