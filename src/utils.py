import re


def extract_user_message(text: str) -> str:
    pattern = r"<\|start_header_id\|>user<\|end_header_id\|>(.*?)<\|eot_id\|>"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""
