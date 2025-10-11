from llama_index.embeddings.huggingface_api import (
    HuggingFaceInferenceAPIEmbedding,
)
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
from config import settings as config
from langchain_openai.chat_models import ChatOpenAI
import logging


logger = logging.getLogger(__name__)


def create_model_embedding():
    embedding_llm = HuggingFaceInferenceAPIEmbedding(
        model_name=config.HUGGINGFACE_MODEL_EMBEDDING,
        token=config.HUGGINGFACE_TOKEN,
    )

    logger.info("Created HuggingFace embedding model")

    return embedding_llm


def create_model_llm(
    temperature: float = config.TEMPERATURE,
    max_new_tokens: int = config.MAX_NEW_TOKENS,
):
    llm = ChatOpenAI(
        name="openrouter",
        base_url="https://openrouter.ai/api/v1",
        model=config.OPENROUTER_MODEL,
        api_key=config.OPENROUTER_API_KEY,
        temperature=temperature,
        max_completion_tokens=max_new_tokens,
        top_p=config.TOP_P,
    )

    logger.info("Created OpenRouter LLM model")
    return llm


def create_mode_llm_huggingface(
    temperature: float = config.TEMPERATURE,
    max_new_tokens: int = config.MAX_NEW_TOKENS,
):
    llm = HuggingFaceInferenceAPI(
        model_name=config.HUGGINGFACE_MODEL_LLM,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        top_p=config.TOP_P,
        token=config.HUGGINGFACE_TOKEN,
    )

    logger.info("Created HuggingFace LLM model")
    return llm
