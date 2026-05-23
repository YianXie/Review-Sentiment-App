import pandas as pd  # type: ignore
import re


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", "", text)
    return text.split()


def main():
    df = pd.read_excel("data/xlsx/Sentiments Dataset.xlsx")
    df = df[df["Sentiment"].isin(["Positive", "Negative"])]

    positive_words = set()
    negative_words = set()
    for _, row in df.iterrows():
        review = clean_text(row["Review"])
        sentiment = row["Sentiment"]

        for word in review:
            if sentiment == "Positive":
                positive_words.add(word)
            else:
                negative_words.add(word)

    while True:
        review = input("Enter a review, or -1 to quit: ").lower().strip()
        if review == "-1":
            break

        num_positive = num_negative = 0
        for word in review.split():
            if word in positive_words:
                num_positive += 1
            elif word in negative_words:
                num_negative += 1

        print(f"Positive Words: {num_positive}")
        print(f"Negative Words: {num_negative}")
        print(f"Unknown Words: {len(review.split()) - num_positive - num_negative}")
        if num_positive > num_negative:
            print("Positive")
        elif num_positive < num_negative:
            print("Negative")
        else:
            print("Neutral")


if __name__ == "__main__":
    main()
