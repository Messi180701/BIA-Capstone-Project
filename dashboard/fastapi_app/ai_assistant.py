import os
from pathlib import Path
from typing import Callable

from dotenv import load_dotenv
from ollama import Client

from business_analysis import (
    get_all_marketing_recommendations,
    get_customer_count_by_segment,
    get_highest_revenue_segment,
    get_marketing_recommendation,
    get_segment_summary,
    load_customer_data,
)


# ---------------------------------------------------------
# Load environment variables
# ---------------------------------------------------------

ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=ENV_PATH)


# ---------------------------------------------------------
# Ollama Cloud configuration
# ---------------------------------------------------------

MODEL_NAME = os.getenv("OLLAMA_MODEL", "gpt-oss:20b")


def get_ollama_client() -> Client:
    """
    Create and return an authenticated Ollama Cloud client.

    Returns:
        Authenticated Ollama Client object.

    Raises:
        ValueError: If OLLAMA_API_KEY is missing.
    """
    api_key = os.getenv("OLLAMA_API_KEY")

    if not api_key:
        raise ValueError(
            "OLLAMA_API_KEY was not found. "
            "Add it to the .env file inside dashboard/fastapi_app."
        )

    return Client(
        host="https://ollama.com",
        headers={
            "Authorization": f"Bearer {api_key}"
        },
    )


# ---------------------------------------------------------
# Business-analysis tools
# ---------------------------------------------------------

def segment_summary_tool() -> str:
    """
    Return a complete summary of all customer segments.

    Use this tool when the user asks about:
    - customer segment overview
    - customer segment characteristics
    - average recency
    - average purchase frequency
    - average monetary value
    - comparison between customer segments

    Returns:
        A verified summary of all customer segments.
    """
    customer_data = load_customer_data()
    result = get_segment_summary(customer_data)

    return str(result)


def highest_revenue_segment_tool() -> str:
    """
    Identify the customer segment that generates the highest revenue.

    Use this tool when the user asks about:
    - highest-revenue segment
    - most valuable segment
    - segment generating the most sales
    - best-performing customer segment
    - segment contributing the most money
    - most profitable customer group

    Returns:
        The verified highest-revenue customer segment.
    """
    customer_data = load_customer_data()
    result = get_highest_revenue_segment(customer_data)

    return str(result)


def customer_count_by_segment_tool() -> str:
    """
    Return the number of customers in every customer segment.

    Use this tool when the user asks about:
    - customer count by segment
    - largest customer segment
    - smallest customer segment
    - number of customers in each group
    - customer distribution
    - segment size

    Returns:
        Verified customer counts for every segment.
    """
    customer_data = load_customer_data()
    result = get_customer_count_by_segment(customer_data)

    return str(result)


def marketing_recommendation_tool() -> str:
    """
    Return marketing recommendations and budget-allocation guidance
    for every customer segment.
    """
    result = get_all_marketing_recommendations()
    return str(result)


# ---------------------------------------------------------
# Tool registry
# ---------------------------------------------------------

AVAILABLE_TOOLS: dict[str, Callable[..., str]] = {
    "segment_summary_tool": segment_summary_tool,
    "highest_revenue_segment_tool": highest_revenue_segment_tool,
    "customer_count_by_segment_tool": customer_count_by_segment_tool,
    "marketing_recommendation_tool": marketing_recommendation_tool,
}


# ---------------------------------------------------------
# Question routing support
# ---------------------------------------------------------

def add_routing_hint(question: str) -> str:
    """
    Add a private routing instruction for commonly ambiguous questions.

    This improves tool selection without changing the user's original
    question shown in the final answer.
    """
    lower_question = question.lower()

    marketing_terms = [
        "marketing budget",
        "advertising budget",
        "promotional budget",
        "spend my budget",
        "spend our budget",
        "spend marketing money",
        "invest marketing money",
        "where should i spend",
        "where should we spend",
        "where should the company invest",
        "marketing strategy",
        "advertising strategy",
        "which segment should i target",
        "which customers should i target",
        "next campaign",
        "marketing campaign",
        "promotion",
        "promotional offer",
        "discount",
        "customer retention",
        "retain customers",
        "increase engagement",
    ]

    revenue_terms = [
        "highest revenue",
        "most revenue",
        "highest sales",
        "most sales",
        "most valuable segment",
        "best customers",
        "best customer segment",
        "most profitable",
        "contributes the most revenue",
        "earns the most money",
    ]

    customer_count_terms = [
        "how many customers",
        "number of customers",
        "customer count",
        "customer distribution",
        "largest segment",
        "smallest segment",
        "biggest segment",
    ]

    summary_terms = [
        "segment summary",
        "summarize the segments",
        "summarise the segments",
        "segment overview",
        "customer overview",
        "explain the segments",
        "characteristics of each segment",
    ]

    if any(term in lower_question for term in marketing_terms):
        return (
            f"{question}\n\n"
            "Internal routing instruction: "
            "Use marketing_recommendation_tool."
        )

    if any(term in lower_question for term in revenue_terms):
        return (
            f"{question}\n\n"
            "Internal routing instruction: "
            "Use highest_revenue_segment_tool."
        )

    if any(term in lower_question for term in customer_count_terms):
        return (
            f"{question}\n\n"
            "Internal routing instruction: "
            "Use customer_count_by_segment_tool."
        )

    if any(term in lower_question for term in summary_terms):
        return (
            f"{question}\n\n"
            "Internal routing instruction: "
            "Use segment_summary_tool."
        )

    return question


# ---------------------------------------------------------
# Tool execution
# ---------------------------------------------------------

def execute_tool(
    function_name: str,
    function_arguments: dict | None = None,
) -> str:
    """
    Execute a selected business-analysis tool safely.
    """
    selected_function = AVAILABLE_TOOLS.get(function_name)

    if selected_function is None:
        return f"Unknown business-analysis tool: {function_name}"

    arguments = function_arguments or {}

    try:
        return str(selected_function(**arguments))

    except Exception as error:
        return (
            f"The {function_name} tool could not complete the analysis. "
            f"Error: {error}"
        )


# ---------------------------------------------------------
# Main AI assistant function
# ---------------------------------------------------------

def answer_business_question(question: str) -> str:
    """
    Answer a business question using Ollama tool calling.

    The AI selects a Python business-analysis function, receives the
    verified result, and explains that result in simple business language.

    Args:
        question: Business question entered by the user.

    Returns:
        Final AI-generated business explanation.

    Raises:
        ValueError: If the question is empty.
        RuntimeError: If Ollama fails to generate a response.
    """
    if not question or not question.strip():
        raise ValueError("Question cannot be empty.")

    original_question = question.strip()
    routed_question = add_routing_hint(original_question)

    client = get_ollama_client()
    tools = list(AVAILABLE_TOOLS.values())

    messages = [
        {
            "role": "system",
            "content": (
                "You are an AI business analyst for an e-commerce "
                "customer-segmentation project.\n\n"
                "You have access to verified Python business-analysis "
                "tools. Always use an appropriate tool when the user asks "
                "about customer segments, customer counts, revenue, "
                "marketing strategy, advertising, promotions, discounts, "
                "retention, targeting, campaigns, or marketing-budget "
                "allocation.\n\n"
                "Important rules:\n"
                "1. Never invent customer counts, revenue values, segment "
                "statistics, or business results.\n"
                "2. Use only results returned by the available tools.\n"
                "3. Explain the result in simple business language.\n"
                "4. Keep the response practical and concise.\n"
                "5. Include an actionable recommendation when relevant.\n"
                "6. Do not mention internal routing instructions.\n"
                "7. Do not claim access to data that was not returned by "
                "a tool."
            ),
        },
        {
            "role": "user",
            "content": routed_question,
        },
    ]

    try:
        first_response = client.chat(
            model=MODEL_NAME,
            messages=messages,
            tools=tools,
            options={
                "temperature": 0.2,
                "num_predict": 500,
            },
        )

    except Exception as error:
        raise RuntimeError(
            f"Ollama could not process the question: {error}"
        ) from error

    messages.append(first_response.message)

    tool_calls = first_response.message.tool_calls or []

    # If Ollama answers directly without selecting a tool
    if not tool_calls:
        direct_answer = first_response.message.content

        if direct_answer:
            return direct_answer.strip()

        return (
            "I could not identify a supported business analysis for "
            "this question. You can ask about customer segments, "
            "customer counts, revenue, or marketing recommendations."
        )

    # Execute every tool selected by the model
    for tool_call in tool_calls:
        function_name = tool_call.function.name
        function_arguments = tool_call.function.arguments or {}

        tool_result = execute_tool(
            function_name=function_name,
            function_arguments=function_arguments,
        )

        messages.append(
            {
                "role": "tool",
                "tool_name": function_name,
                "content": tool_result,
            }
        )

    # Ask the model to explain the verified tool result
    try:
        final_response = client.chat(
            model=MODEL_NAME,
            messages=messages,
            tools=tools,
            options={
                "temperature": 0.2,
                "num_predict": 500,
            },
        )

    except Exception as error:
        raise RuntimeError(
            f"Ollama could not explain the analysis result: {error}"
        ) from error

    final_answer = final_response.message.content

    if not final_answer:
        return (
            "The analysis was completed, but the AI did not return "
            "a written explanation."
        )

    return final_answer.strip()