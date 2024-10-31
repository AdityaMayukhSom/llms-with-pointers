import math
from turtle import distance
from typing import List


def kullback_leibler_divergence(P: List[float], Q: List[float], epsilon: float = 1e-8) -> float:
    total = 0
    for i in range(len(P)):
        total += P[i] * math.log((P[i] + epsilon) / (Q[i] + epsilon))
    return total


def jensen_shannon_divergence(P: List[float], Q: List[float], epsilon: float = 1e-8) -> float:
    M = [0] * len(P)

    for i in range(len(P)):
        M[i] = 0.5 * (P[i] + Q[i])

    kl_PM = kullback_leibler_divergence(P, M, epsilon)
    kl_QM = kullback_leibler_divergence(Q, M, epsilon)

    divergence = 0.5 * (kl_PM + kl_QM)
    return divergence


if __name__ == "__main__":
    MAX_PAD_WIDTH = 30

    P = [0.36, 0.48, 0.16]
    Q = [0.30, 0.50, 0.20]

    kld = kullback_leibler_divergence(P, Q)
    jsd = jensen_shannon_divergence(P, Q)

    print("Kullback Leibler Divergence".ljust(MAX_PAD_WIDTH), kld)
    print("Jensen Shannon Divergence".ljust(MAX_PAD_WIDTH), jsd)
    print("Jensen Shannon Distance".ljust(MAX_PAD_WIDTH), math.sqrt(jsd))
