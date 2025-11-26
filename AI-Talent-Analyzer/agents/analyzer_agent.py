from .base_agent import BaseAgent

import re
from dateutil import parser

def extract_years_from_text(text: str) -> float:
    """
    Rule based calculation for years of experience to avoid hallucination 
    """
    if not text:
        return 0.0

    patterns = [
        r"([A-Za-z]{3,9} \d{4})\s*[–-]\s*([A-Za-z]{3,9} \d{4})",    # Jan 2020 – May 2021
    ]

    total_months = 0

    for pattern in patterns:
        for start, end in re.findall(pattern, text):
            try:
                s = parser.parse(start)
                e = parser.parse(end)
                diff = (e.year - s.year) * 12 + (e.month - s.month)
                print('diff: ',diff)
                if 0 < diff < 600:  # sanity check
                    total_months += diff
            except:
                pass

    years = round(total_months / 12, 1)

    return years


class AnalyzerAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="Analyzer",
            instructions="""Analyze candidate profiles and extract:
            1. Technical skills (as a list)
            2. Years of experience (numeric)
            3. Education level
            4. Experience level (Junior/Mid-level/Senior)
            5. Key achievements
            6. Domain expertise
            Format the output as structured data.""",
        )

    async def run(self, messages):
        print("Analyzer: Analyzing candidate profile")

        extracted_data = eval(messages[-1]["content"])

        analysis_prompt = f"""
            From the following structured resume data AND the raw resume text, extract a
            COMPLETE and EXHAUSTIVE list of ALL technical skills, tools, libraries, 
            frameworks, programming languages, ML/CV/AI techniques, robotics skills, 
            embedded systems skills, cloud skills, DevOps technologies, software tools, and
            domain keywords.

            SCAN THE ENTIRE RESUME:
            - skills section
            - work experience
            - projects
            - research
            - summary
            - links
            - tools mentioned anywhere in text

            Your goal is to capture **EVERY SKILL AND TECHNICAL KEYWORD** the candidate knows.

            RETURN EXACT JSON in this structure:

            {{
                "technical_skills": [
                    "python", "c++", "ros2", "pytorch", "tensorflow",
                    "opencv", "machine learning", "deep learning", "sql",
                    ...
                ],
                "years_of_experience": number,
                "education": [
                    {{
                        "degree": "Bachelors/Masters/PhD",
                        "field": "",
                        "institution": "",
                        "year": ""
                    }}]
                "experience_level": "Junior/Mid-level/Senior",
                "key_achievements": [],
                "domain_expertise": [
                    "robotics", "computer vision", "autonomous systems",
                    ...
                ]
            }}

            VERY IMPORTANT:
            - Normalize all skill names to lowercase.
            - Deduplicate keywords.
            - Include synonyms only once (e.g., “machine learning” not both “ML” and “machine learning”).
            - Include all tools: cloud, ML, CV, robotics, embedded, devops, software, libraries, hardware, OS, frameworks.
            - Keep only true skills or technologies (no soft skills like communication).
            - Extract skills even if mentioned briefly inside experience descriptions.

            Resume structured data:
            {extracted_data["structured_data"]}

            Raw resume text:
            {extracted_data["raw_text"]}

            Return ONLY the JSON object. No explanation.
        """

        analysis_results = self._query_ollama(analysis_prompt)
        parsed = self._parse_json_safely(analysis_results)

        if "error" in parsed:
            parsed = {
                "technical_skills": [],
                "years_of_experience": 0,
                "education": {"level": "Unknown", "field": "Unknown"},
                "experience_level": "Junior",
                "key_achievements": [],
                "domain_expertise": [],
            }

        # confidence logic
        skills = parsed.get("technical_skills", [])
        # years = parsed.get("years_of_experience", 0)
        edu = parsed.get("education", [])

        raw_text = extracted_data.get("raw_text", "")
        years = extract_years_from_text(raw_text)
        parsed["years_of_experience"] = 4


        # a. skills score
        if len(skills) == 0:
            skills_score = 0.2
        elif len(skills) < 5:
            skills_score = 0.6
        else:
            skills_score = 1.0

        # b. experience score

        try:
            years = float(years) if years is not None else 0
        except:
            years = 0

        experience_score = 1.0 if years > 0 else 0.3

        # # c. education score
        valid_degrees = ["bachelor", "bachelor's", "master", "master's", "phd", "doctor"]
        if isinstance(edu, list) and edu:
            level_ok_list = []
            field_ok_list = []

            for study in edu:
                degree_text = str(study.get("degree", "")).lower()
                field_text = str(study.get("field", "")).strip()

                level_ok = any(k in degree_text for k in valid_degrees)
                field_ok = field_text not in ["Unknown", "", "unknown", "n/a", "N/A", None]

                level_ok_list.append(level_ok)
                field_ok_list.append(field_ok)

            if all(level_ok_list) and all(field_ok_list):
                education_score = 1.0
            elif any(level_ok_list) or any(field_ok_list):
                education_score = 0.5
            else:
                education_score = 0.2

        
        # Combination
        confidence = round(
            (skills_score + experience_score + education_score) / 3, 2)
        # print(f"skills_score: {skills_score}, experience_score: {experience_score}, education_score: {education_score}. Final confidence: {confidence}")        
    
        result = {
            "skills_analysis": parsed,
            "analysis_timestamp": "2024-03-14",
            "confidence_score": confidence,
        }

        return result


