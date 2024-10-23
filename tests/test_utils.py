import unittest

from src.constants import INSTRUCT_PROMPT_TEMPLATE
from src.utils import extract_user_message


class TestExtractUserMessage(unittest.TestCase):
    def test_extraction(self):
        user_original_message = "this is user"
        message = INSTRUCT_PROMPT_TEMPLATE.format(
            system_message="this is system",
            user_message=user_original_message,
        )
        user_extracted_message = extract_user_message(message)
        print(user_extracted_message)
        self.assertEqual(user_original_message, user_extracted_message)
