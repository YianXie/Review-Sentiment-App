import math


def tf(t: str, d: list[str]) -> float:
    return d.count(t) / len(d)


def idf(t: str, D: list[list[str]]) -> float:
    count = 0
    for d in D:
        if t in d:
            count += 1
    return math.log10(len(D) / count)


def tf_idf(t: str, d: list[str], D: list[list[str]]) -> float:
    return tf(t, d) * idf(t, D)
