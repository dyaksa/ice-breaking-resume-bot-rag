from langchain_community.document_loaders import PyPDFLoader
from langchain.schema import SystemMessage, HumanMessage
from module.llm_interface import create_model_llm
from pydantic import BaseModel, Field
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


class ProfileData(BaseModel):
    name: str = Field("", description="Full name of the individual")
    current_position: str = Field("", description="Current job title and company")
    location: str = Field("", description="Location of the individual")
    summary: str = Field("", description="Professional summary or bio")
    experiences: list = Field([], description="List of professional experiences")
    education: list = Field([], description="List of educational qualifications")
    skills: list = Field([], description="List of skills")
    certifications: list = Field([], description="List of certifications")
    languages: list = Field([], description="List of languages known")
    interests: list = Field([], description="List of personal interests")


def extract_profile_pdf(pdf_path: str) -> Dict[str, Any]:
    try:
        loader = PyPDFLoader(pdf_path)
        loaded = loader.load()

        document_text = "\n".join([doc.page_content for doc in loaded])

        llm_model = create_model_llm(
            temperature=0.1,
            max_new_tokens=500,
        )

        system_message = SystemMessage(
            content="""
            You are an expert in talent acquisition and recruitment. Extract structured information from the provided resume PDF content. Focus on aspects for efficient resume retrieval.
            Extract the following fields: name, current_position, location, summary, experiences, education, skills, certifications, languages, interests.
            """
        )

        user_message = HumanMessage(
            content=f"""
            from the following resume content, extract the relevant information based on initial document.
            {document_text}
            """
        )

        oneshot_example = HumanMessage(
            content="""
        Generate sub-queries based on this initial resume content.
        John Doe
        Software Engineer at Tech Solutions
        San Francisco, CA
        Experienced Software Engineer with a demonstrated history of working in the information technology and services industry. Skilled in Python, Java, and cloud computing. Strong engineering professional with a Bachelor's degree in Computer Science from State University.
        Experience:
        - Software Engineer at Tech Solutions (2018 - Present)
        - Developed and maintained web applications using Python and Java.
        - Collaborated with cross-functional teams to define project requirements and deliverables.
        - Junior Developer at Web Innovations (2016 - 2018)
        - Assisted in the development of client websites and applications.
        - Participated in code reviews and team meetings.
        Education:
        - Bachelor of Science in Computer Science, State University (2012 - 2016)
        Skills:
        - Programming Languages: Python, Java, JavaScript
        - Frameworks: Django, React
        - Tools: Git, Docker, AWS
        Certifications:
        - AWS Certified Solutions Architect
        Languages:
        - English (Native)
        - Spanish (Professional Proficiency)
        Interests:
        - Hiking, Photography, Traveling
        """
        )

        llm_invoke = [system_message, oneshot_example, user_message]
        response = llm_model.invoke(llm_invoke)
        return response.content
    except Exception as e:
        logger.error(f"Error extracting profile from PDF: {e}")
        return {}
