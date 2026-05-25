from django.shortcuts import render

from review_sentiment_app.predict import predict


def index(request):
    review = ""
    prediction = None

    if request.method == "POST":
        review = request.POST.get("review", "").strip()
        if review:
            prediction = predict(review)

    return render(
        request,
        "sentiment/index.html",
        {"review": review, "prediction": prediction},
    )
