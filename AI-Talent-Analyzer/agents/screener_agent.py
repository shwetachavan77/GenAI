from .base_agent import BaseAgent
import json


class ScreenerAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="Screener",
            instructions="""Screen candidates based on:
                - Qualification alignment
                - Experience relevance
                - Skill match percentage
                - Cultural fit indicators
                - Red flags or concerns
                Provide comprehensive screening reports.
            """,
        )

    def compute_role_specific_score(self, role, skills):

        role = (role or "").lower()
        s = set([x.lower() for x in skills])

        ROLE_TABLE = {
            "robotics": {
                "must": ["c++", "ros", "ros2", "linux", "opencv", "robotics", "slam"],
                "good": ["pytorch", "kalman", "ekf", "pid", "motion planning"],
            },
            "machine learning": {
                "must": ["python", "pytorch", "tensorflow", "machine learning"],
                "good": ["huggingface", "mlops", "docker", "aws"],
            },
            "cv engineer": {
                "must": ["opencv", "pytorch", "computer vision", "image processing"],
                "good": ["yolo", "segmentation", "detection"],
            },
            "nlp": {
                "must": ["nlp", "transformers", "huggingface", "python"],
                "good": ["lora", "openai api"],
            },
            "backend": {
                "must": ["python", "sql", "docker", "rest", "apis"],
                "good": ["aws", "redis", "kafka"],
            },
            "embedded": {
                "must": ["c", "c++", "embedded", "linux", "microcontroller"],
                "good": ["uart", "spi", "i2c", "rtos"],
            },
            "devops": {
                "must": ["docker", "kubernetes", "linux", "ci/cd"],
                "good": ["aws", "terraform"],
            },
        }

        # No role mapped -> neutral score
        if role not in ROLE_TABLE:
            return 60.0

        must = ROLE_TABLE[role]["must"]
        good = ROLE_TABLE[role]["good"]

        must_hits = sum(1 for m in must if m in s)
        good_hits = sum(1 for g in good if g in s)

        must_score = (must_hits / len(must)) * 70
        good_score = (good_hits / len(must)) * 30

        return round(must_score + good_score, 2)

    def compute_screener_score(self, context):
        """Computes all scores and returns final screener summary"""

        analysis = context.get("analysis_results", {})
        if isinstance(analysis, str):
            try:
                analysis = json.loads(analysis)
            except:
                analysis = {}
                print("No analysis result extracted")

        skills_analysis = analysis.get("skills_analysis", {})
        skills = skills_analysis.get("technical_skills", [])
        years = skills_analysis.get("years_of_experience", 0)
        edu = skills_analysis.get("education", [])
        exp_level = skills_analysis.get("experience_level", "Mid-level")
        analyzer_conf = analysis.get("confidence_score", 0)

        job_matches = context.get("job_matches", {}).get("matched_jobs", [])
        best_match = job_matches[0]["match_score"] if job_matches else 0
        role = job_matches[0]["title"] if job_matches else "general"

        # a. experience fit score (0-100)
        years = None
        try:
            years = context["analysis_results"]["skills_analysis"].get("years_of_experience")
        except:
            years = None

        # normalize to convert None or invalid types to 0
        if not isinstance(years, (int, float)):
            years = 0

        exp_fit = 100 if years >= 3 else (60 if years >= 1 else 30)

        # b. education score (0-100)
        education_score = 80
        if isinstance(edu, list) and edu:
            for e in edu:
                deg = str(e.get("degree", "")).lower()
                if any(k in deg for k in ["phd", "master"]):
                    education_score = 100
                elif "bachelor" in deg:
                    education_score = 80

        # c. Role relevance (0-100)
        role_score = self.compute_role_specific_score(role, skills)

        # weighted score
        final_score = (
            best_match * 0.30 +
            analyzer_conf * 100 * 0.25 +
            education_score * 0.15 +
            exp_fit * 0.10 +
            role_score * 0.20
        )

        final_score = round(final_score, 2)

        return {
            "final_score": final_score,
            "experience_fit": exp_fit,
            "education_score": education_score,
            "role_score": role_score,
            "analyzer_confidence": round(analyzer_conf * 100, 2),
            "best_job_match": best_match,
            "computed_role": role,
        }
    
    # llm summary
    def generate_llm_summary(self, context, role):
        summary_prompt = f"""
            You are a senior recruiter.

            Write a SHORT screening summary for:

            ROLE: {role}

            Candidate context:
            {json.dumps(context, indent=2)}

            Write:
            - 3-5 strengths
            - 2-3 weaknesses
            - a brief verdict (Strong / Medium / Weak fit)

            No JSON. Just clean text.
        """

        return self._query_ollama(summary_prompt)

    async def run(self, messages):
        print("👥 Screener: Conducting initial screening")

        raw = messages[-1]["content"]

        try:
            context = json.loads(raw)
        except:
            context = eval(raw)

        score_blob = self.compute_screener_score(context)

        role = score_blob.get("computed_role", "general")
        llm_summary = self.generate_llm_summary(context, role)

        return {
            # "screening_report": llm_summary,    
            "screening_score": score_blob,
            "screening_summary": llm_summary,
            "screening_timestamp": "2024-03-14",
        }
