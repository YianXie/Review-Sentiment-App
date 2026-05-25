import pandas as pd  # type: ignore
from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
from sklearn.model_selection import train_test_split  # type: ignore
from sklearn.naive_bayes import MultinomialNB  # type: ignore
from sklearn.metrics import classification_report  # type: ignore


def main() -> None:
    df = pd.read_excel("data/xlsx/Customer_Sentiment.xlsx")

    X: list[str] = []
    y: list[str] = []
    for _, row in df.iterrows():
        X.append(row["review_text"])
        y.append(row["sentiment"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0
    )

    vectorizer = TfidfVectorizer()
    X_train_tfidf = vectorizer.fit_transform(X_train)

    clf = MultinomialNB()
    clf.fit(X_train_tfidf, y_train)

    X_test_tfidf = vectorizer.transform(X_test)
    y_pred = clf.predict(X_test_tfidf)

    print(classification_report(y_test, y_pred))

    while True:
        review = input("Enter a review, or -1 to quit: ").strip()
        if review == "-1":
            break

        review_tfidf = vectorizer.transform([review])  # note the list wrapping
        prediction = clf.predict(review_tfidf)
        print(f"Sentiment: {prediction[0]}")

        probs = clf.predict_proba(review_tfidf)[0]
        for label, prob in zip(clf.classes_, probs):
            print(f"  {label}: {prob:.4f}")


if __name__ == "__main__":
    main()
