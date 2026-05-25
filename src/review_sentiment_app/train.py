import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report


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

    joblib.dump(vectorizer, "models/vectorizer.joblib")
    joblib.dump(clf, "models/clf.joblib")


if __name__ == "__main__":
    main()
