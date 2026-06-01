from pathlib import Path
from typing import Literal
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import joblib

MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "models"

tokenizer = BertTokenizer.from_pretrained(MODELS_DIR / "bert_sentiment")
model = BertForSequenceClassification.from_pretrained(MODELS_DIR / "bert_sentiment")
ID2LABEL = joblib.load(MODELS_DIR / "id2label.joblib")

model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


def predict(review: str) -> Literal["positive", "negative", "neutral"]:
    inputs = tokenizer(
        review, return_tensors="pt", truncation=True, padding=True, max_length=128
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    pred_id = outputs.logits.argmax(dim=-1).item()
    return ID2LABEL[pred_id]
