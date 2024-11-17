import os
from multiprocessing import Pool
from typing import Iterable, Literal, Tuple

import nltk
import pandas as pd
from loguru import logger
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from rouge_score import rouge_scorer

TypeScore = dict[Literal["rouge1", "rouge2", "rougeL"], rouge_scorer.scoring.Score]


def create_bleu_input(target: str, prediction: str):
    return [word_tokenize(prediction)], word_tokenize(target)


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
    # https://corejava25hours.com/2024/06/15/9-a-automatic-evaluation-metrics-bleu-rouge-meteor/
    # nltk.download("punkt_tab")

    rouge_types = ["rouge1", "rouge2", "rougeL"]
    scorer = rouge_scorer.RougeScorer(rouge_types=rouge_types, use_stemmer=True)
    metric_count = 10  # 3 for ROUGE-1, 3 for ROUGE-2, 3 for ROUGE-L and 1 for BLEU

    base_dir = files[0]
    abs_file_name = files[1]
    ori_file_name = files[2]
    abs_file_path = os.path.abspath(os.path.join(base_dir, abs_file_name))
    ori_file_path = os.path.abspath(os.path.join(base_dir, ori_file_name))
    invalid_result = [abs_file_name] + [-1] * metric_count

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

        bleu_original, bleu_abstract = create_bleu_input(original, abstract)
        smooth = SmoothingFunction().method4
        bleu_score = sentence_bleu(bleu_original, bleu_abstract, smoothing_function=smooth)

        return [
            abs_file_name,
            rouge1.precision,
            rouge1.recall,
            rouge1.fmeasure,
            rouge2.precision,
            rouge2.recall,
            rouge2.fmeasure,
            rougeL.precision,
            rougeL.recall,
            rougeL.fmeasure,
            bleu_score,
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
        "BLEU Score",
    ]

    in_dir = "./results_test"
    out_file = "./rouge_data/results.csv"
    pool_chunk_size = 1024

    os.makedirs(os.path.dirname(out_file), exist_ok=True)

    pool = Pool()
    res = pool.imap(compute_rouge, get_result_files(in_dir), chunksize=pool_chunk_size)
    df = pd.DataFrame(res, columns=columns)
    df.to_csv(out_file)
