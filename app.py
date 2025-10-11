from module.extract_profile_pdf import extract_profile_pdf
from module.data_processing import (
    split_profile_data,
    create_vector_index,
    verify_embedding_model,
)
from fastapi import FastAPI
from module.query_engine import generate_facts_candidate, answer_user_question
from config import settings as config
import gradio as gr
import uuid
import logging
import os

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)

active_session = {}

logger = logging.getLogger(__name__)


def process_resume(file):
    temp_file_path = None

    try:
        if file is None:
            return "No file uploaded", None

        # Handle different Gradio file upload formats
        if hasattr(file, "name"):
            # Gradio >= 4.0 returns a file object with .name attribute
            document_path = file.name
            logger.info(f"Using uploaded file path: {document_path}")
        elif isinstance(file, str):
            # Sometimes Gradio returns a file path string
            document_path = file
            logger.info(f"Using file path string: {document_path}")

        logger.info(f"Processing resume at path: {document_path}")

        profile_data = extract_profile_pdf(document_path)

        if not profile_data:
            logger.error("No profile data extracted from PDF")
            return "No profile data extracted from PDF", None

        nodes = split_profile_data(profile_data)
        if not nodes:
            logger.error("No data chunks created from profile data")
            return "No data chunks created from profile data", None

        index = create_vector_index(nodes)
        if not index:
            logger.error("Failed to create vector index")
            return "Failed to create vector index", None

        if not verify_embedding_model(index):
            logger.error("Embedding model verification failed")
            return "Embedding model verification failed", None

        initial_facts = generate_facts_candidate(index)

        session_id = str(uuid.uuid4())

        active_session[session_id] = index

        return initial_facts, session_id

    except Exception as e:
        logger.error(f"Error processing resume: {e}")
        return f"Error processing resume: {str(e)}", [], []

    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
                logger.info(f"Cleaned up temporary file: {temp_file_path}")
            except Exception as cleanup_error:
                logger.warning(
                    f"Failed to clean up temporary file {temp_file_path}: {cleanup_error}"
                )


def handle_chat(message, chat_history, session_id=None):
    try:
        if not message.strip():
            return chat_history, ""

        if session_id is None:
            error_msg = "Please upload and process a resume first."
            chat_history.append((message, error_msg))
            return chat_history, ""

        # Get the most recent index (in a real app, you'd want to track sessions properly)
        index = active_session[session_id]

        # Generate response using the query engine
        response = answer_user_question(index, message)

        chat_history.append((message, response))
        return chat_history, ""

    except Exception as e:
        logger.error(f"Error in chat handler: {e}")
        error_msg = f"Error processing your question: {str(e)}"
        chat_history.append((message, error_msg))
        return chat_history, ""


def create_gradio_interface():
    with gr.Blocks() as demo:
        gr.Markdown("# Resume Chatbot")
        gr.Markdown("Upload a resume PDF and ask questions about the candidate.")
        with gr.Row():
            with gr.Column():
                pdf_input = gr.File(label="Upload Resume PDF", file_types=[".pdf"])
                facts_result = gr.Textbox(
                    label="Initial Facts about the Candidate", lines=10
                )
                upload_button = gr.Button("Process Resume")
                session_id = gr.Textbox(label="Session ID", visible=False)
            with gr.Column():
                chat_output = gr.Chatbot(height=500)
                user_input = gr.Textbox(
                    label="Your Question",
                    placeholder="Ask a question about the candidate...",
                )
                send_button = gr.Button("Send")

        upload_button.click(
            fn=process_resume,
            inputs=[pdf_input],
            outputs=[facts_result, session_id],
        )

        send_button.click(
            fn=handle_chat,
            inputs=[user_input, chat_output, session_id],
            outputs=[chat_output, user_input],
        )

        # Also allow pressing Enter in the text input to send message
        user_input.submit(
            fn=handle_chat,
            inputs=[user_input, chat_output],
            outputs=[chat_output, user_input],
        )

        return demo


app = FastAPI(title="Resume Chatbot")

app = gr.mount_gradio_app(
    app=app, blocks=create_gradio_interface(), path="/", server_port=config.PORT
)
