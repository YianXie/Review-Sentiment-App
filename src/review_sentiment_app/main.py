from review_sentiment_app.predict import predict


def main() -> None:
    while True:
        review = input("Enter a review, or -1 to quit: ").strip()
        if review == "-1":
            break

        prediction = predict(review)
        print(f"Sentiment: {prediction}")


if __name__ == "__main__":
    main()
