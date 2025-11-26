# AI Talent Analyzer Engine

A lightweight, local-LLM multi-agent system that analyzes resumes end-to-end `extraction -> skill analysis -> job matching -> screening -> final recommendation` wrapped in a clean Streamlit UI

Runs fully offline using **Ollama + LLaMA 3.2**

---

## Features
- **PDF Resume Extraction** 
- **Skill & Experience Analysis** 
- **Hybrid Job Matching Engine**  
- **Candidate Screening** 
- **Final Recommendation Engine** 
- **Streamlit UI** with 4 tabs:
  1. `Analysis`
  2. `Job Matches`
  3. `Screening`
  4. `Recommendation` 

---

<video width="400" controls>
  <source src="demo/demo.mp4" type="video/mp4">
  Your browser does not support HTML video.
</video>


---

## Architecture Overview
The system uses a modular multi-agent pipeline:

- **ExtractorAgent** — Extracts text + contact info from resumes  
- **AnalyzerAgent** — Parses skills, experience, and education
- **MatcherAgent** — Matches candidates to jobs using 80% LLM semantic score + 20% fuzzy & keyword logic  
- **ScreenerAgent** — Scores experience fit & role suitability  
- **RecommenderAgent** — Generates a final recommendation summary  
- **OrchestratorAgent** — Coordinates the entire pipeline  

All agents return **strict JSON** to ensure reliability

---

## Tech Stack
- Python 3.12  
- Streamlit  
- SQLite (for job database)  
- Ollama / LLaMA 3.2  
- Regular expressions and datetime utilities  

