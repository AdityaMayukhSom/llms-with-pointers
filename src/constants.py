import enum


class MessageTemplates:
    INSTRUCT_PROMPT = """\
    <|begin_of_text|>
    <|start_header_id|>system<|end_header_id|>
    {system_message}
    <|eot_id|>
    <|start_header_id|>user<|end_header_id|>
    {user_message}
    <|eot_id|>
    <|start_header_id|>assistant<|end_header_id|>"""

    SYSTEM_MESSAGE = """\
    You are an expert in reading articles and coming up with brief, yet clear and informative abstracts.
    Only return a generated abstract. Do not provide any explanations or unnecessary words. 
    Do not provide anything which is not necessary or related to the abstract. Just provide the abstract."""

    USER_MESSAGE = """\
    Summarize this following article under {max_words} words:

    {article}"""


@enum.unique
class DataFilePaths(str, enum.Enum):
    VAL_INDX = "single/tfindex/val.tfindex"
    TEST_INDX = "single/tfindex/test.tfindex"
    TRAIN_INDX = "single/tfindex/train.tfindex"

    VAL_DATA = "single/tfrecord/val.tfrecord"
    TEST_DATA = "single/tfrecord/test.tfrecord"
    TRAIN_DATA = "single/tfrecord/train.tfrecord"


@enum.unique
class DataPointKeys(str, enum.Enum):
    ARTICLE = "article"
    ABSTRACT = "abstract"
    PROMPT = "prompt"
