import re

def extract_certifications(text):

    text = text.replace("\r", "\n")

    patterns = [
        r"certifications?(.*?)(education|experience|projects|skills|achievements|internships|languages|interests|$)",
        r"certificates?(.*?)(education|experience|projects|skills|achievements|internships|languages|interests|$)"
    ]

    certifications = []

    for pattern in patterns:

        match = re.search(
            pattern,
            text,
            re.IGNORECASE | re.DOTALL
        )

        if match:

            section = match.group(1).strip()

            lines = section.split("\n")

            for line in lines:

                line = line.strip()

                if len(line) > 2:
                    certifications.append(line)

            break

    return list(set(certifications))