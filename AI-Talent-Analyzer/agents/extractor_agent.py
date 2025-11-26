import re
import json
from pdfminer.high_level import extract_text
from .base_agent import BaseAgent


class ExtractorAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="Extractor",
            instructions="Extract raw text + contact info + high-level structure."
        )

    def extract_contact_info(self, text):
        """extract name, email, phone, location using regex + heuristics."""

        email = re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text)
        phone = re.search(r"(\+?\d[\d\-()\s]{7,}\d)", text)

        lines = text.split("\n")
        name = None
        location = None

        if len(lines) >= 1:
            first = lines[0].strip()
            if 2 <= len(first.split()) <= 10: 
                name = first

        # Search for common city/state/country patterns
        location_regex = r"(New York|San Francisco|Los Angeles|Pittsburgh|Atlanta|Chicago|Boston|London|Toronto|Bangalore|Mumbai|Pune|Delhi|Paris|Berlin|Singapore|Tokyo|Sydney)"
        for line in lines[:10]:  # Only search in top 10 lines
            loc_match = re.search(location_regex, line, re.IGNORECASE)
            if loc_match:
                location = loc_match.group(0)
                break

        return {
            "name": name or "Not specified",
            "email": email.group(0) if email else "Not specified",
            "phone": phone.group(0) if phone else "Not specified",
            "location": location or "Not specified",
        }

    async def run(self, messages):
        """process a resume and extract info"""

        print("Extractor: Processing Resume")

        resume_data = messages[-1]["content"]

        if isinstance(resume_data, str) and resume_data.startswith("{"):
            resume_data = json.loads(resume_data)

        if resume_data.get("file_path"):
            raw_text = extract_text(resume_data["file_path"])
        else:
            raw_text = resume_data.get("text", "")

        contact = self.extract_contact_info(raw_text)

        result = {
            "raw_text": raw_text,
            "contact_info": contact,
            "structured_data": {}  
        }

        return result
