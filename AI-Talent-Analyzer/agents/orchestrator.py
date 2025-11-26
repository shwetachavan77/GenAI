from .base_agent import BaseAgent
from .extractor_agent import ExtractorAgent
from .analyzer_agent import AnalyzerAgent
from .matcher_agent import MatcherAgent
from .screener_agent import ScreenerAgent
from .recommender_agent import RecommenderAgent
import streamlit as st
import json

status = st.empty()

class OrchestratorAgent(BaseAgent):
    def __init__(self, status_box, progress_bar):
        super().__init__(
            name="Orchestrator",
            instructions="""Coordinate the recruitment workflow and delegate tasks to specialized agents.
            Ensure proper flow of information between extraction, analysis, matching, screening, and recommendation phases.
            Maintain context and aggregate results from each stage.""",
        )
        self.status_box = status_box
        self.progress_bar = progress_bar
        self._setup_agents()

    def _setup_agents(self):
        self.extractor = ExtractorAgent()
        self.analyzer = AnalyzerAgent()
        self.matcher = MatcherAgent()
        self.screener = ScreenerAgent()
        self.recommender = RecommenderAgent()

    async def run(self, messages):
        prompt = messages[-1]["content"]
        response = self._query_ollama(prompt)
        return self._parse_json_safely(response)

    async def process_application(self, resume_data):
        print("Orchestrator: Starting application process")

        workflow_context = {
            "resume_data": resume_data,
            "status": "initiated",
            "current_stage": "extraction",
        }

        try:
            self.status_box.write("Processing...")
            self.progress_bar.progress(10)
            # Extract resume information
            extracted_data = await self.extractor.run(
                [{"role": "user", "content": json.dumps(resume_data)}]
            )
            workflow_context.update(
                {"extracted_data": extracted_data, "current_stage": "analysis"}
            )
            print("Extractor completed")
            self.status_box.write("Extractor Completed. Starting Analyzer...")
            self.progress_bar.progress(20)

            # Analyze candidate profile
            analysis_results = await self.analyzer.run(
                [{"role": "user", "content": json.dumps(extracted_data)}]
            )
            workflow_context.update(
                {"analysis_results": analysis_results, "current_stage": "matching"}
            )
            print("Analyzer completed")
            self.status_box.write("Analyzer completed. Starting Matcher...")
            self.progress_bar.progress(40)

            # Match with jobs
            job_matches = await self.matcher.run(
                [{"role": "user", "content": json.dumps(analysis_results)}]
            )
            workflow_context.update(
                {"job_matches": job_matches, "current_stage": "screening"}
            )
            print("Matcher completed")
            self.status_box.write("Matcher completed. Started Screener...")
            self.progress_bar.progress(60)

            # Screen candidate
            screening_results = await self.screener.run(
                [{"role": "user", "content": json.dumps(workflow_context)}]
            )
            workflow_context.update(
                {
                    "screening_results": screening_results,
                    "current_stage": "recommendation",
                }
            )
            print("Screener completed")
            self.status_box.write("Screener completed. Started Recommender...")
            self.progress_bar.progress(80)

            # Generate recommendations
            final_recommendation = await self.recommender.run(
                [{"role": "user", "content": json.dumps(workflow_context)}]
            )
            workflow_context.update(
                {"final_recommendation": final_recommendation, "status": "completed"}
            )
            print("Recommender completed")
            self.status_box.write("Recommender completed. Generating report...")
            self.progress_bar.progress(95)

            return workflow_context

        except Exception as e:
            workflow_context.update({"status": "failed", "error": str(e)})
            raise