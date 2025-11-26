from .base_agent import BaseAgent


class RecommenderAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="Recommender",
            instructions="""
                Generate final recommendations considering:
                1. Extracted profile
                2. Skills analysis
                3. Job matches
                4. Screening results
                Provide clear next steps and specific recommendations.
                Additionally, return a confidence score from 0-100.
                Do not share any opinion or other thinking process.

                HARD CONSTRAINTS:
                - NEVER describe the input JSON
                - Be to the point and concise
                - Act like a recommendation guru and only give feedback.

            """

        )

    async def run(self, messages):
        print("Recommender: Generating final recommendations")

        workflow_context = eval(messages[-1]["content"])

        skills_conf = workflow_context["analysis_results"]["confidence_score"]  # 0-1
        best_job_match = max(
            [job["match_score"] for job in workflow_context["job_matches"]["matched_jobs"]],
            default=0
        )
        screening_score = workflow_context["screening_results"]["screening_score"]["final_score"]
        
        # Dynamic confidence level
        final_confidence = (
            (skills_conf * 100 * 0.4) +    
            (best_job_match * 0.4) +        
            (screening_score * 0.2)         
        )
        final_confidence = round(final_confidence, 2)
        print(final_confidence)

        if final_confidence >= 85:
            confidence_label = "high"
        elif final_confidence >= 60:
            confidence_label = "medium"
        else:
            confidence_label = "low"

        recommendation = self._query_ollama(str(workflow_context))

        return {
            "final_recommendation": recommendation,
            "recommendation_timestamp": "2025-03-14",
            "confidence_level": confidence_label,
            "confidence_score": final_confidence
        }
