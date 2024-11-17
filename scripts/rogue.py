import os
from multiprocessing import Pool
from typing import Iterable, Literal, Tuple

import pandas as pd
from loguru import logger
from rouge_score import rouge_scorer

TypeScore = dict[Literal["rouge1", "rouge2", "rougeL"], rouge_scorer.scoring.Score]


def get_result_files(base_dir: str) -> Iterable[Tuple[str, str, str]]:
    SUF_ART = "_article.txt"
    SUF_ABS = "_gen_abstract.txt"
    SUF_ORI = "_ori_abstract.txt"

    for filename in os.listdir(base_dir):
        if os.path.isdir(os.path.join(base_dir, filename)):
            continue
        if not filename.endswith(SUF_ART):
            continue

        prefix = filename.split(SUF_ART)[0]
        abs_file = prefix + SUF_ABS
        ori_file = prefix + SUF_ORI
        yield base_dir, abs_file, ori_file

    logger.success("exhausted files in path: {}".format(base_dir))


def compute_rouge(files: Tuple[str, str, str]):
    rouge_types = ["rouge1", "rouge2", "rougeL"]
    scorer = rouge_scorer.RougeScorer(rouge_types=rouge_types, use_stemmer=True)

    base_dir = files[0]
    abs_file_path = os.path.abspath(os.path.join(base_dir, files[1]))
    ori_file_path = os.path.abspath(os.path.join(base_dir, files[2]))
    invalid_result = [abs_file_path] + [-1] * 8

    try:
        abs_file = open(abs_file_path, "r", encoding="UTF-8")
        abstract = abs_file.read()
        abs_file.close()
    except FileNotFoundError as e:
        logger.exception("file not found")
        return invalid_result

    try:
        ori_file = open(ori_file_path, "r", encoding="UTF-8")
        original = ori_file.read()
        ori_file.close()
    except FileNotFoundError as e:
        logger.exception("file not found")
        return invalid_result

    try:
        scores: TypeScore = scorer.score(original, abstract)
        rouge1 = scores["rouge1"]
        rouge2 = scores["rouge2"]
        rougeL = scores["rougeL"]

        return [
            abs_file_path,
            rouge1.precision,
            rouge1.recall,
            rouge1.fmeasure,
            rouge2.precision,
            rouge2.recall,
            rouge2.fmeasure,
            rougeL.precision,
            rougeL.recall,
            rougeL.fmeasure,
        ]
    except ValueError as e:
        logger.error("invalid rouge type found")
        return invalid_result


if __name__ == "__main__":
    columns = [
        "File",
        "ROUGE-1 Precision",
        "ROUGE-1 Recall",
        "ROUGE-1 F-Measure",
        "ROUGE-2 Precision",
        "ROUGE-2 Recall",
        "ROUGE-2 F-Measure",
        "ROUGE-L Precision",
        "ROUGE-L Recall",
        "ROUGE-L F-Measure",
    ]

    in_dir = "./results_test/generated_v2"
    out_file = "generated_v2.csv"

    pool = Pool()
    res = pool.imap(compute_rouge, get_result_files("./results_test/generated_v2"), chunksize=1024)
    df = pd.DataFrame(res, columns=columns)
    df.to_csv("generated_v2.csv")
