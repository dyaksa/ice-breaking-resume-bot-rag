from module.extract_profile_pdf import extract_profile_pdf
from module.data_processing import (
    split_profile_data,
    create_vector_index,
    verify_embedding_model,
)
from module.query_engine import generate_facts_candidate, answer_user_question
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)

logger = logging.getLogger(__name__)


def process_resume(document_path: str):
    try:
        logger.info(f"Processing resume at path: {document_path}")

        profile_data = extract_profile_pdf(document_path)

        if not profile_data:
            logger.error("No profile data extracted from PDF")
            return

        nodes = split_profile_data(profile_data)
        if not nodes:
            logger.error("No data chunks created from profile data")
            return

        index = create_vector_index(nodes)
        if not index:
            logger.error("Failed to create vector index")
            return

        if not verify_embedding_model(index):
            logger.error("Embedding model verification failed")
            return

        initial_facts = generate_facts_candidate(index)
        logger.info(f"Initial facts about candidate: {initial_facts}")

        chatbot_interface(index)

    except Exception as e:
        logger.error(f"Error extracting profile data: {e}")
        return


def chatbot_interface(index):
    print("Welcome to the Resume Chatbot! Type 'exit' to quit.")

    while True:
        user_question = input("You: ")
        if user_question.lower() == "exit":
            print("Exiting the chatbot. Goodbye!")
            break

        answer = answer_user_question(index, user_question)
        print(f"Bot: {answer}")


def main():
    document_path = "cv_dyaksa_jauharddin_nour_02-06-2025.pdf"
    index = process_resume(document_path)
    if index:
        chatbot_interface(index)


if __name__ == "__main__":
    main()
