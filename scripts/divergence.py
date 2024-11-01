import math
from typing import List

from tabulate import tabulate


def kullback_leibler(P: List[float], Q: List[float], epsilon: float = 1e-8):
    total = 0.0
    for i in range(len(P)):
        total += P[i] * math.log((P[i] + epsilon) / (Q[i] + epsilon))
    return total


def jensen_shannon(P: List[float], Q: List[float]):
    M = [0] * len(P)

    for i in range(len(P)):
        M[i] = 0.5 * (P[i] + Q[i])

    kl_PM = kullback_leibler(P, M)
    kl_QM = kullback_leibler(Q, M)

    divergence = 0.5 * (kl_PM + kl_QM)
    return divergence


if __name__ == "__main__":
    # Path to the input file. The first line should contain two
    # numbers: N and M. N represents the number of examples to
    # process, and M indicates the number of probabilities in
    # each distribution, P and Q. For each example, there will
    # be two lines: the first line contains M numbers for the
    # probabilities in P, and the second line contains M numbers
    # for the probabilities in Q. Therefore, the file will have
    # a total of 2 * N lines following the first line.
    in_path = "./temp/Divergence_Input.txt"

    # Path to store the results of the calculation.
    out_path = "./temp/Divergence_Output.txt"

    in_file = open(in_path, mode="r", encoding="UTF-8")
    out_file = open(out_path, mode="w", encoding="UTF-8")

    table = []
    table_headers = [
        "Kullback Leibler Divergence",
        "Jensen Shannon Divergence",
        "Jensen Shannon Distance",
    ]

    N, M = list(map(int, in_file.readline().split()))

    for i in range(N):
        P = list(map(float, in_file.readline().split()))
        Q = list(map(float, in_file.readline().split()))

        assert len(P) == M and len(Q) == M

        kld = kullback_leibler(P, Q)
        jsd = jensen_shannon(P, Q)

        table.append([kld, jsd, math.sqrt(jsd)])

    table_str = tabulate(
        table,
        headers=table_headers,
        showindex="always",
        tablefmt="pretty",
    )

    print(table_str, file=out_file)

    in_file.close()
    out_file.close()
