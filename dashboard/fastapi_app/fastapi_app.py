from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from ai_assistant import answer_business_question


app = FastAPI(
    title="E-commerce Customer Intelligence API",
    version="1.0.0",
)


class BusinessQuestionRequest(BaseModel):
    question: str


@app.get("/")
def home():
    return {"message": "Customer Intelligence API is running"}


@app.post("/ai-business-assistant")
def ai_business_assistant(request: BusinessQuestionRequest):
    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="The question cannot be empty.",
        )

    try:
        answer = answer_business_question(question)

        return {
            "question": question,
            "answer": answer,
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Unable to generate an answer: {error}",
        ) from error