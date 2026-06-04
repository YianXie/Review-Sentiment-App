import nltk
import regex
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download("punkt_tab")
nltk.download("stopwords")


def clean_review(review: str) -> str:
    clean = review.lower()
    clean = regex.sub(r"\p{Extended_Pictographic}", "", clean)
    clean = regex.sub(r"[^\w\s]", "", clean)
    clean = regex.sub(r"http\S+", "", clean)

    stop_words = set(stopwords.words("english"))
    clean_words = word_tokenize(clean)
    clean_words = [word for word in clean_words if word not in stop_words]

    return clean
