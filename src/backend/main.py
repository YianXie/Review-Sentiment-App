from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from review_sentiment_app.predict import predict


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/get-review-sentiment/{review}")
def get_review_sentiment(review: str):
    return predict(review)
