import pandas as pd
import joblib
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "models"
LABEL2ID = {"negative": 0, "neutral": 1, "positive": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


class ReviewDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt",
        )
        self.labels = torch.tensor([LABEL2ID[label] for label in labels])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels": self.labels[idx],
        }


def main():
    df = pd.read_excel("data/xlsx/Customer_Sentiment.xlsx")
    X = df["review_text"].tolist()
    y = df["sentiment"].tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0
    )

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=3, id2label=ID2LABEL, label2id=LABEL2ID
    )

    train_dataset = ReviewDataset(X_train, y_train, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=2e-5)

    # Fine-tuning loop
    model.train()
    for epoch in range(3):  # 3 epochs is usually enough
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            outputs = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                labels=batch["labels"].to(device),
            )
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1} — Loss: {total_loss / len(train_loader):.4f}")

    # Evaluate
    model.eval()
    test_dataset = ReviewDataset(X_test, y_test, tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=32)
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in test_loader:
            outputs = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            )
            preds = outputs.logits.argmax(dim=-1).cpu().tolist()
            all_preds.extend([ID2LABEL[p] for p in preds])
            all_labels.extend([ID2LABEL[label.item()] for label in batch["labels"]])

    print(classification_report(all_labels, all_preds))

    # Save
    model.save_pretrained(MODELS_DIR / "bert_sentiment")
    tokenizer.save_pretrained(MODELS_DIR / "bert_sentiment")
    joblib.dump(ID2LABEL, MODELS_DIR / "id2label.joblib")


if __name__ == "__main__":
    main()
