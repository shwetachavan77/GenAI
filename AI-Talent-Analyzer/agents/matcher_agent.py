import json
import re
import sqlite3
from difflib import SequenceMatcher
from .base_agent import BaseAgent
from db.database import JobDatabase


class MatcherAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="Matcher",
            instructions="""Match candidate profiles with job positions.
            Use hybrid scoring:
            - LLM semantic comparison
            - fuzzy string similarity
            - keyword overlap
            Return detailed match scores with reasons."""
        )
        self.db = JobDatabase()

    def extract_json_block(self, text):
        """Extract first valid JSON dict/list from messy LLM output."""
        text = text.replace("```json", "").replace("```", "").strip()

        # Try objects {} and arrays []
        patterns = [r"\{[\s\S]*?\}", r"\[[\s\S]*?\]"]

        for p in patterns:
            matches = re.findall(p, text)
            for m in matches:
                try:
                    return json.loads(m)
                except:
                    continue

        return None

    def llm_match_score(self, ollama_call, candidate_skills, job_requirements):
        prompt = f"""
            You must output ONE AND ONLY ONE JSON OBJECT. 
            NEVER output lists, arrays, multiple JSON blocks, code snippets, markdown fences, or explanations.

            Your OUTPUT MUST MATCH EXACTLY this schema:

            {{
            "match_score": <integer between 0 and 100>,
            "reason": "<single-line string>"
            }}

            RULES:
            - match_score MUST be an INTEGER, NOT float.
            - reason MUST be a SINGLE STRING, NOT array.
            - DO NOT include backticks.
            - DO NOT include multiple JSON objects.
            - DO NOT include any text outside the JSON.
            - DO NOT include comments or explanations.
            - match_score must ALWAYS be a whole number between 0 and 100.
            - If unsure, guess.

        """

        raw = ollama_call(prompt)
        parsed = self.extract_json_block(raw)

        if parsed is None:
            return 0, "No JSON found"

        # If the LLM returned a JSON string
        if isinstance(parsed, str):
            return 0, parsed

        # If the LLM returned a list of dicts -> pick first dict
        if isinstance(parsed, list):
            if len(parsed) == 0:
                return 0, "Empty JSON list"
            parsed = parsed[0]

        if not isinstance(parsed, dict):
            return 0, "Malformed JSON"

        # Extract fields safely
        score = parsed.get("match_score", 0)
        reason = parsed.get("reason", "No explanation")

        if isinstance(reason, list):
            reason = "; ".join(str(r) for r in reason)

        return int(score), reason

    # Fuzzy Similarity
    def fuzzy_similarity(self, a, b):
        return SequenceMatcher(None, a.lower(), b.lower()).ratio()

    # Hybrid Score
    def hybrid_score(self, llm_func, candidate_skills, job_requirements):
        
        raw_llm_score, llm_reason = self.llm_match_score(
            llm_func,
            candidate_skills,
            job_requirements
        )

        # normalize LLM: 0-100 -> 0-1
        llm_norm = raw_llm_score / 100.0 if raw_llm_score > 1 else raw_llm_score
        llm_norm = max(0.0, min(llm_norm, 1.0))                                     # clamp
        
        fuzzy_matches = []
        for cs in candidate_skills:
            best = max([self.fuzzy_similarity(cs, jr) for jr in job_requirements] or [0])
            fuzzy_matches.append(best)

        # avg fuzzy in 0-1 space
        fuzzy_norm = sum(fuzzy_matches) / max(1, len(job_requirements))
        fuzzy_norm = max(0.0, min(fuzzy_norm, 1.0))                                 # clamp

        req_set = set([r.lower() for r in job_requirements])
        skill_set = set([s.lower() for s in candidate_skills])

        overlap = len(req_set & skill_set)                                          # exact matches
        keyword_norm = overlap / max(1, len(req_set))                               # 0-1

        # hybrid
        final_norm = (
            0.80 * llm_norm +
            0.10 * fuzzy_norm +
            0.10 * keyword_norm
        )
        final_score = int(final_norm * 100)

        return final_score, int(llm_norm * 100), int(fuzzy_norm * 100), llm_reason

    async def run(self, messages):
        print("Matcher: Matching Resume with available jobs")
        raw = messages[-1].get("content", "{}")

        try:
            data = self._parse_json_safely(raw)
        except:
            print("Could not parse input to matcher")
            return {"matched_jobs": []}

        skills_analysis = data.get("skills_analysis")
        if not skills_analysis:
            print("No skills_analysis found.")
            return {"matched_jobs": []}

        # Normalize skills
        candidate_skills = [s.lower().strip() for s in skills_analysis.get("technical_skills", [])]

        # level = skills_analysis.get("experience_level", "Mid-level").capitalize()
        raw_level = None
        if isinstance(skills_analysis, dict):
            raw_level = skills_analysis.get("experience_level")

        level = (raw_level or "Mid-level")
        level = str(level).strip().capitalize()

        jobs = self.search_jobs(candidate_skills, level)
        print("The experience level is: ")
        print(skills_analysis.get("experience_level", "No level found, going to look for mid level jobs"))

        all_matches = []

        for job in jobs:
            reqs = [r.lower() for r in job["requirements"]]

            final_score, llm_s, fuzzy_s, reason = self.hybrid_score(
                self._query_ollama, candidate_skills, reqs
            )

            if final_score >= 40:
                all_matches.append({
                    "title": job["title"],
                    "company": job["company"],
                    "match_score": final_score,
                    "llm_score": llm_s,
                    "fuzzy_score": fuzzy_s,
                    "reason": reason,
                    "location": job["location"],
                    "requirements": job["requirements"]
                })

        all_matches.sort(key=lambda x: x["match_score"], reverse=True)

        return {
            "matched_jobs": all_matches,
            "count": len(all_matches)
        }

    def search_jobs(self, skills, experience_level):
        lvl = (experience_level or "").strip().lower()
        if "junior" in lvl:
            lvl_norm = "Junior"
        elif "mid" in lvl:
            lvl_norm = "Mid-level"
        elif "senior" in lvl:
            lvl_norm = "Senior"
        else:
            lvl_norm = None

        def run_query(with_level):
            base = "SELECT * FROM jobs"
            params = []
            where = []

            if with_level and lvl_norm:
                where.append("experience_level = ?")
                params.append(lvl_norm)

            if skills:
                skill_clauses = []
                for s in skills:
                    skill_clauses.append("requirements LIKE ?")
                    params.append(f"%{s}%")

                where.append("(" + " OR ".join(skill_clauses) + ")")

            if where:
                base += " WHERE " + " AND ".join(where)

            with sqlite3.connect(self.db.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cur = conn.cursor()
                cur.execute(base, params)
                rows = cur.fetchall()

            return [
                {
                    "id": row["id"],
                    "title": row["title"],
                    "company": row["company"],
                    "location": row["location"],
                    "type": row["type"],
                    "experience_level": row["experience_level"],
                    "salary_range": row["salary_range"],
                    "description": row["description"],
                    "requirements": json.loads(row["requirements"]),
                    "benefits": json.loads(row["benefits"]) if row["benefits"] else [],
                }
                for row in rows
            ]

        jobs = run_query(with_level=bool(lvl_norm))
        if jobs:
            return jobs

        print("No jobs found with level filter. Retrying without level...")
        return run_query(with_level=False)
