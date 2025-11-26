import streamlit as st
import asyncio
import os
from datetime import datetime
from pathlib import Path
from streamlit_option_menu import option_menu
from agents.orchestrator import OrchestratorAgent
from utils.logger import setup_logger

st.set_page_config(
    page_title="Talent Analyzer Engine",
    page_icon="👽",
    layout="wide",
    initial_sidebar_state="expanded",
)

logger = setup_logger()

st.markdown(
    """
    <style>
        .stProgress .st-bo {
            background-color: #00a0dc;
        }
        .success-text {
            color: #00c853;
        }
        .warning-text {
            color: #ffd700;
        }
        .error-text {
            color: #ff5252;
        }
        .st-emotion-cache-1v0mbdj.e115fcil1 {
            border: 1px solid #ddd;
            border-radius: 10px;
            padding: 20px;
        }
    </style>
""",
    unsafe_allow_html=True,
)


async def process_resume(file_path, status_box, progress_bar) :
    upload_msg = st.empty()
    upload_msg.info("Resume uploaded successfully! Processing...")
    await asyncio.sleep(10)
    upload_msg.empty()
    try:
        orchestrator = OrchestratorAgent(status_box, progress_bar)
        resume_data = {
            "file_path": file_path,
            "submission_timestamp": datetime.now().isoformat(),
        }
        return await orchestrator.process_application(resume_data)
    except Exception as e:
        logger.error(f"Error processing resume: {str(e)}")
        raise


def save_uploaded_file(uploaded_file):
    """Save uploaded file and return the file path"""
    try:
        save_dir = Path("uploads")
        save_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = save_dir / f"resume_{timestamp}_{uploaded_file.name}"

        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        return str(file_path)
    except Exception as e:
        st.error(f"Error saving file: {str(e)}")
        raise


def main():
    with st.sidebar:
        st.title("AI Talent Analyzer ✨")
        selected = option_menu(
            menu_title="Navigation",
            options=["Upload Resume", "About"],
            icons=["cloud-upload", "info-circle"],
            menu_icon="cast",
            default_index=0,
        )

    if selected == "Upload Resume":
        st.header("Resume Analysis")
        st.write("Upload a resume to get AI-powered insights and job matches")

        uploaded_file = st.file_uploader(
            "Choose a PDF resume file",
            type=["pdf"],
            help="Upload a PDF resume to analyze",
        )

        if uploaded_file:
            try:
                with st.spinner("Saving uploaded file..."):
                    file_path = save_uploaded_file(uploaded_file)

                progress_bar = st.progress(0)
                status_box = st.empty()

                try:

                    result = asyncio.run(process_resume(file_path, status_box, progress_bar))
                    if result["status"] == "completed":
                        progress_bar.progress(100)
                        status_box.text("Analysis complete!")

                        tab1, tab2, tab3, tab4 = st.tabs(
                            [
                                "🧠 Analysis",
                                "💼 Job Matches",
                                "📋 Screening",
                                "🎓 Recommendation",
                            ]
                        )

                        with tab1:
                            st.subheader("Skills Analysis")
                            st.write(result["analysis_results"]["skills_analysis"])
                            st.metric(
                                "Confidence Score",
                                f"{result['analysis_results']['confidence_score']:.0%}",
                            )
  
                        with tab2:
                            st.subheader("Matched Positions")

                            matches = result["job_matches"]["matched_jobs"]
                            if not matches:
                                st.warning("No suitable positions found.")

                            seen_titles = set()
                            for job in matches:
                                if job["title"] in seen_titles:
                                    continue
                                seen_titles.add(job["title"])

                                with st.container():
                                    col1, col2, col3 = st.columns([2, 1, 1])

                                    with col1:
                                        st.write(f"**{job['title']}**")
                                    with col2:
                                        st.write(f"Match: {job.get('match_score', 'N/A')}%")
                                    with col3:
                                        st.write(f"📍 {job.get('location', 'N/A')}")

                                with st.expander("View job details"):
                                    st.json(job)
                                st.divider()

                        with tab3:
                            st.subheader("Screening Results")
                            st.metric(
                                "Screening Score",
                                f"{result['screening_results']['screening_score']['final_score']}%",
                            )
                            st.write(result["screening_results"]["screening_score"])
                            st.write(result["screening_results"]["screening_summary"])
                        

                        with tab4:
                            st.subheader("Final Recommendation")
                            st.info(
                                result["final_recommendation"]["final_recommendation"],
                                icon="💡",
                            )

                        output_dir = Path("results")
                        output_dir.mkdir(exist_ok=True)
                        output_file = (
                            output_dir
                            / f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                        )

                        with open(output_file, "w") as f:
                            f.write(str(result))

                        st.success(f"Results saved to: {output_file}")

                    else:
                        st.error(
                            f"Process failed at stage: {result.get('current_stage', 'N/A')}\n"
                            f"Error: {result.get('error', 'Unknown error')}"
                        )

                except Exception as e:
                    st.error(f"Error processing resume: {str(e)}")
                    logger.error(f"Processing error: {str(e)}", exc_info=True)

                finally:
                    try:
                        os.remove(file_path)
                    except Exception as e:
                        logger.error(f"Error removing temporary file: {str(e)}")

            except Exception as e:
                st.error(f"Error handling file upload: {str(e)}")
                logger.error(f"Upload error: {str(e)}", exc_info=True)

    elif selected == "About":
        st.write("""
            ### **👽 About Multi-Agent Talent Analyzer Engine**

            A lightweight, AI-powered system that uses coordinated agents to analyze resumes and give clear, structured insights.  
            Built with:

            - **Ollama (llama3.2)** for fast, local LLM processing  
            - **Multi-agent** based orchestration  
            - **Streamlit** for a clean, simple interface  

            What the engine does:
            1. 📄 Extracts key information from resumes  
            2. 📊 Analyzes candidate strengths and experience  
            3. 📁 Matches them to relevant job roles  
            4. 🎯 Performs quick screening  
            5. 🚀 Provides actionable recommendations  

            Upload a resume and let the agents handle the rest ✨.
        """)



if __name__ == "__main__":
    main()