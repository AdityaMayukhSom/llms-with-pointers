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
    Summarize the following article in a single paragraph, strictly within {max_words} words. Focus only on the main facts, key points, and essential details from the article, avoiding any unnecessary details, commentary, or opinions. Do not begin with introductory phrases like 'Here is a summary of the article,' and ensure the summary does not exceed {max_words} words. Provide only the abstract text in a concise, fact-focused format, directly reflecting the article's content without extra explanations. The generated summary should be a single paragraph that captures the main points clearly and precisely and should be factually correct according to the article provided.
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
