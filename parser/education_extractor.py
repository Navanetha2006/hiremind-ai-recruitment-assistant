import re

def extract_education(text):

    text = text.lower()

    education_patterns = [

        # Engineering
        "b.tech",
        "btech",
        "b.e",
        "m.e",
        "m.tech",
        "mtech",

        # Science
        "b.sc",
        "bsc",
        "m.sc",
        "msc",

        # Commerce
        "b.com",
        "bcom",
        "m.com",
        "mcom",

        # Management
        "mba",
        "bba",

        # Computer Applications
        "bca",
        "mca",

        # Medicine
        "mbbs",
        "bds",
        "md",

        # Research
        "phd",
        "doctorate",

        # Full Forms
        "bachelor of technology",
        "master of technology",
        "bachelor of engineering",
        "master of engineering",
        "bachelor of science",
        "master of science",
        "bachelor of commerce",
        "master of commerce",
        "bachelor of business administration",
        "master of business administration"
    ]

    found = []

    for edu in education_patterns:

        if edu in text:
            found.append(edu)

    return list(set(found))