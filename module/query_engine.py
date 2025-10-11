from llama_index.core import VectorStoreIndex, PromptTemplate
from module.llm_interface import create_model_llm
from config import settings as config
from typing import Any
import logging


logger = logging.getLogger(__name__)


def generate_facts_candidate(index: VectorStoreIndex) -> str:
    try:
        model_llm = create_model_llm(
            temperature=0.1,
            max_new_tokens=500,
        )

        prompt_template = """
            You are an expert in talent recruitment. Analyze the following resume profile.
            context information is below:

            {context_str}

            into 4-5 more focused aspects to facilitate the resume search process.
            Only use the information provided and do not create your own requirements.
            If you don't know the answer, just say that you don't know, do not try to make up an answer.
            provide a detailed answer about the candidate.
        """

        facts_template = PromptTemplate(template=prompt_template)

        query_engine = index.as_query_engine(
            streaming=False,
            similarity_top_k=config.SIMILARITY_TOP_K,
            llm=model_llm,
            text_qa_template=facts_template,
        )

        query = (
            "Provide a detailed analysis of the candidate based on the profile data."
        )

        result = query_engine.query(query)
        return result.response
    except Exception as e:
        logger.error(f"Error creating LLM model: {e}")
        return "Error creating LLM model"


def answer_user_question(index: VectorStoreIndex, question: str) -> Any:
    try:
        model_llm = create_model_llm(
            temperature=config.TEMPERATURE,
            max_new_tokens=config.MAX_NEW_TOKENS,
        )

        base_retriever = index.as_retriever(
            similarity_top_k=config.SIMILARITY_TOP_K,
        )
        nodes = base_retriever.retrieve(question)

        context_str = "\n\n".join([node.node.get_text() for node in nodes])

        prompt_template = f"""
        You are an expert in recruiting talent who helps determine the best candidates on resume profiles.
        Use the following information to answer the question.
        You must provide a detailed explanation of the profile.
        context information is below:
        {context_str}

        If you don't know the answer, just say you don't know, don't try to make up a false answer.
        """

        user_question_template = PromptTemplate(template=prompt_template)

        query_engine = index.as_query_engine(
            streaming=False,
            similarity_top_k=config.SIMILARITY_TOP_K,
            llm=model_llm,
            text_qa_template=user_question_template,
        )

        result = query_engine.query(question)
        return result.response
    except Exception as e:
        logger.error(f"Error creating LLM model: {e}")
        return "Error creating LLM model"
