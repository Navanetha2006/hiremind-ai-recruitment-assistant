import re


def extract_experience(text):

    matches = re.findall(
        r"(\d+)\+?\s*years",
        text.lower()
    )

    return matches