import math
import pandas as pd  # type: ignore
import re

from review_sentiment_app.tf_idf import tf_idf  # type: ignore


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", "", text)
    return text.split()


def get_review_probs(
    review: str,
    words: dict[str, dict[str, float]],
    overall_positive_prob: float,
    overall_negative_prob: float,
    positive_words_weights: float,
    negative_words_weights: float,
    vocab_size: int,
):
    review_positive_prob = review_negative_prob = 0.0
    for word in review.split():
        review_positive_prob += math.log10(
            (words.get(word, {"positive": 0})["positive"] + 1)
            / (positive_words_weights + vocab_size)
        )
        review_negative_prob += math.log10(
            (words.get(word, {"negative": 0})["negative"] + 1)
            / (negative_words_weights + vocab_size)
        )

    return (
        review_positive_prob + math.log10(overall_positive_prob),
        review_negative_prob + math.log10(overall_negative_prob),
    )


def main():
    df = pd.read_excel("data/xlsx/Sentiments Dataset.xlsx")
    df = df[df["Sentiment"].isin(["Positive", "Negative"])].head(100)

    words = {}
    num_positive_reviews = num_negative_reviews = 0
    positive_words_weights = negative_words_weights = 0.0
    vocabularies = set()

    word_matrix: list[list[str]] = []
    for rowIndex, (_, row) in enumerate(df.iterrows()):
        review = clean_text(row["Review"])
        word_matrix.append(review)
        for word in review:
            if word not in words:
                words[word] = {"positive": 0.0, "negative": 0.0}
            vocabularies.add(word)
        if row["Sentiment"] == "Positive":
            num_positive_reviews += 1
        else:
            num_negative_reviews += 1

    for rowIndex, row in enumerate(word_matrix):
        for word in row:
            weight = tf_idf(word, row, word_matrix)
            if df.iloc[rowIndex]["Sentiment"] == "Positive":
                positive_words_weights += weight
                words[word]["positive"] += weight
            else:
                negative_words_weights += weight
                words[word]["negative"] += weight

    overall_positive_prob = num_positive_reviews / len(df)
    overall_negative_prob = num_negative_reviews / len(df)
    while True:
        review = input("Enter a review, or -1 to quit: ").lower().strip()
        if review == "-1":
            break

        positive_prob, negative_prob = get_review_probs(
            review,
            words,
            overall_positive_prob,
            overall_negative_prob,
            positive_words_weights,
            negative_words_weights,
            len(vocabularies),
        )
        print(f"Positive prob: {positive_prob}")
        print(f"Negative prob: {negative_prob}")
        if positive_prob > negative_prob:
            print("Positive")
        elif positive_prob < negative_prob:
            print("Negative")
        else:
            print("Neutral")


if __name__ == "__main__":
    main()
