from typing import Dict, Any, List
from llama_index.core import Document, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from module.llm_interface import create_model_embedding
from config import settings as config
import json
import logging


logger = logging.getLogger(__name__)


def split_profile_data(profile_data: Dict[str, Any] | str) -> List:
    try:
        if isinstance(profile_data, dict):
            profile_data = json.dumps(profile_data)

        logger.info(f"Profile JSON: {profile_data}")

        document = Document(text=profile_data)

        splitter = SentenceSplitter(chunk_size=config.CHUNK_SIZE)

        nodes = splitter.get_nodes_from_documents([document])

        logger.info(f"Split profile data into {len(nodes)} chunks")

        return nodes
    except Exception as e:
        logger.error(f"Error splitting profile data: {e}")
        return []


def create_vector_index(nodes: List) -> VectorStoreIndex:
    try:
        # Get the embedding model
        embedding_model = create_model_embedding()
        # Create a VectorStoreIndex from the nodes
        index = VectorStoreIndex(
            nodes=nodes, embed_model=embedding_model, show_progress=True
        )

        return index
    except Exception as e:
        logger.error(f"Error creating vector index: {e}")
        return None


def verify_embedding_model(index: VectorStoreIndex) -> bool:
    try:
        vector_store = index._storage_context.vector_store

        node_ids = list(index.index_struct.nodes_dict.keys())

        missing_embedding = False

        for node_id in node_ids:
            embedding = vector_store.get(node_id)

            if embedding is None:
                missing_embedding = True
                logger.warning(f"Missing embedding for node ID: {node_id}")
            else:
                logger.info(f"Embedding exists for node ID: {node_id}")

        return not missing_embedding
    except Exception as e:
        logger.error(f"Error verifying embedding model: {e}")
        return False
