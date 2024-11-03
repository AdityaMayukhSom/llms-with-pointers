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
    <|start_header_id|>assistant<|end_header_id|>
    """

    SYSTEM_MESSAGE = """\
    Summarize the following article strictly within {max_words} words. Focus solely on the main points, avoiding unnecessary details or commentary. Do not exceed {max_words} words under any circumstances. Include important facts mentioned in the article. Provide only the abstract text without any extra comments or explanations The output should only contain the summary, and nothing else. 
    """

    USER_MESSAGE = "[text summarization]: {article}"


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
