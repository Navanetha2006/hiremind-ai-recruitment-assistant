SKILL_CATEGORIES = {

    "Software Engineering": [
        "python", "java", "c", "c++", "c#",
        "javascript", "typescript", "php",
        "sql", "html", "css",
        "react", "angular", "vue",
        "django", "flask", "fastapi",
        "node.js", "express"
    ],

    "Data Science": [
        "pandas", "numpy", "matplotlib",
        "seaborn", "statistics",
        "data analysis",
        "data visualization"
    ],

    "AI & Machine Learning": [
        "machine learning",
        "deep learning",
        "tensorflow",
        "pytorch",
        "keras",
        "nlp",
        "computer vision",
        "opencv",
        "generative ai",
        "llm"
    ],

    "Cloud & DevOps": [
        "aws",
        "azure",
        "gcp",
        "docker",
        "kubernetes",
        "terraform",
        "jenkins",
        "linux"
    ],

    "Database": [
        "mysql",
        "postgresql",
        "mongodb",
        "oracle",
        "sqlite"
    ],

    "Cyber Security": [
        "ethical hacking",
        "penetration testing",
        "network security",
        "cyber security",
        "siem"
    ],

    "Mechanical Engineering": [
        "autocad",
        "solidworks",
        "catia",
        "ansys",
        "manufacturing",
        "cnc",
        "thermodynamics"
    ],

    "Civil Engineering": [
        "staad pro",
        "etabs",
        "surveying",
        "construction management",
        "autocad civil"
    ],

    "Electrical Engineering": [
        "power systems",
        "electrical machines",
        "control systems",
        "plc",
        "circuit design"
    ],

    "Electronics": [
        "embedded systems",
        "arduino",
        "raspberry pi",
        "vlsi",
        "verilog",
        "pcb design"
    ],

    "Finance": [
        "financial analysis",
        "accounting",
        "budgeting",
        "investment analysis",
        "taxation"
    ],

    "Marketing": [
        "digital marketing",
        "seo",
        "sem",
        "content marketing",
        "social media marketing",
        "google analytics"
    ],

    "Human Resources": [
        "recruitment",
        "talent acquisition",
        "employee engagement",
        "payroll",
        "hr analytics"
    ],

    "Healthcare": [
        "patient care",
        "clinical research",
        "medical coding",
        "healthcare management"
    ],

    "UI/UX & Design": [
        "figma",
        "adobe xd",
        "photoshop",
        "illustrator",
        "ui design",
        "ux design"
    ],

    "Management": [
        "project management",
        "business analysis",
        "operations management",
        "leadership",
        "strategic planning"
    ]
}


def extract_skills(text):

    text = text.lower()

    found_skills = []

    for category, skills in SKILL_CATEGORIES.items():

        for skill in skills:

            if skill.lower() in text:
                found_skills.append(skill)

    return sorted(list(set(found_skills)))


def detect_domain(text):

    text = text.lower()

    scores = {}

    for category, skills in SKILL_CATEGORIES.items():

        score = 0

        for skill in skills:

            if skill.lower() in text:
                score += 1

        scores[category] = score

    best_domain = max(scores, key=scores.get)

    if scores[best_domain] == 0:
        return "General"

    return best_domain


def extract_skills_by_category(text):

    text = text.lower()

    categorized_skills = {}

    for category, skills in SKILL_CATEGORIES.items():

        matched_skills = []

        for skill in skills:

            if skill.lower() in text:
                matched_skills.append(skill)

        if matched_skills:
            categorized_skills[category] = matched_skills

    return categorized_skills


def extract_technologies(text):

    text = text.lower()

    technology_categories = [

        "Software Engineering",
        "Data Science",
        "AI & Machine Learning",
        "Cloud & DevOps",
        "Database",
        "Cyber Security",
        "Electronics"
    ]

    technologies = []

    for category in technology_categories:

        for skill in SKILL_CATEGORIES[category]:

            if skill.lower() in text:
                technologies.append(skill)

    return sorted(list(set(technologies)))


def extract_programming_languages(text):

    text = text.lower()

    programming_languages = [

        "python",
        "java",
        "c",
        "c++",
        "c#",
        "javascript",
        "typescript",
        "php",
        "sql",
        "html",
        "css"
    ]

    found = []

    for lang in programming_languages:

        if lang in text:
            found.append(lang)

    return sorted(list(set(found)))


def extract_frameworks(text):

    text = text.lower()

    frameworks = [

        "react",
        "angular",
        "vue",
        "django",
        "flask",
        "fastapi",
        "express",
        "tensorflow",
        "pytorch",
        "keras"
    ]

    found = []

    for framework in frameworks:

        if framework in text:
            found.append(framework)

    return sorted(list(set(found)))


def extract_cloud_technologies(text):

    text = text.lower()

    cloud_skills = [

        "aws",
        "azure",
        "gcp",
        "docker",
        "kubernetes",
        "terraform",
        "jenkins"
    ]

    found = []

    for cloud in cloud_skills:

        if cloud in text:
            found.append(cloud)

    return sorted(list(set(found)))