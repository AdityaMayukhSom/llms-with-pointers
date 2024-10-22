import enum


@enum.unique
class DataPointKeys(str, enum.Enum):
    ARTICLE = "article"
    ABSTRACT = "abstract"
