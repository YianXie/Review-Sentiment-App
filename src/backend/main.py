from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from review_sentiment_app.predict import predict


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ReviewRequest(BaseModel):
    review: str


@app.post("/api/get-review-sentiment")
def get_review_sentiment(review_request: ReviewRequest):
    return predict(review_request.review)
