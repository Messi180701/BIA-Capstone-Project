import os
from pathlib import Path

from dotenv import load_dotenv
from ollama import Client

from business_analysis import (
    get_customer_count_by_segment,
    get_highest_revenue_segment,
    get_marketing_recommendation,
    get_segment_summary,
    load_customer_data,
)


ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=ENV_PATH)


def get_ollama_client() -> Client:
    api_key = os.getenv("OLLAMA_API_KEY")

    if not api_key:
        raise ValueError(
            "OLLAMA_API_KEY was not found. Check the .env file."
        )

    return Client(
        host="https://ollama.com",
        headers={
            "Authorization": f"Bearer {api_key}"
        },
    )


def generate_business_explanation(
    question: str,
    verified_result: str,
) -> str:
    client = get_ollama_client()

    prompt = f"""
You are an AI business analyst for an e-commerce customer
segmentation project.

The numerical result below was calculated using verified Python
business logic.

User question:
{question}

Verified business result:
{verified_result}

Explain the result in simple business language.

Rules:
- Do not change or invent any numbers.
- Do not claim access to information not included in the result.
- Keep the response concise and practical.
- Include a useful business recommendation when appropriate.
"""

    response = client.chat(
        model="gpt-oss:20b",
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        options={
            "num_predict": 500,
            "temperature": 0.3,
        },
    )

    return response["message"]["content"]


def answer_business_question(question: str) -> str:
    if not question or not question.strip():
        raise ValueError("Question cannot be empty.")

    cleaned_question = question.strip()
    lower_question = cleaned_question.lower()

    df = load_customer_data()

    if (
        "highest revenue" in lower_question
        or "most revenue" in lower_question
        or "most valuable segment" in lower_question
        or "highest sales" in lower_question
    ):
        result = get_highest_revenue_segment(df)

    elif (
        "customer count" in lower_question
        or "number of customers" in lower_question
        or "how many customers" in lower_question
    ):
        result = get_customer_count_by_segment(df)

    elif (
        "segment summary" in lower_question
        or "summarize segments" in lower_question
        or "overview of segments" in lower_question
    ):
        result = get_segment_summary(df)

    elif (
        "marketing recommendation" in lower_question
        or "marketing strategy" in lower_question
        or "recommend campaign" in lower_question
        or "target customers" in lower_question
    ):
        result = get_marketing_recommendation(df)

    else:
        result = (
            "The available analysis supports customer counts by segment, "
            "segment summaries, highest-revenue segment identification, "
            "and marketing recommendations."
        )

    return generate_business_explanation(
        question=cleaned_question,
        verified_result=str(result),
    )