from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from adv_service.logger import logger



persist_directory = "adv_service/chroma_db"
embeddings_model = OllamaEmbeddings(model="nomic-embed-text")

def load_vector_store():
    """loads vectore store if exists"""
    try:
        logger.info("Попытка загрузить векторное хранилище")
        vector_store = Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings_model,
        )
        logger.info(f"choosen embed model: {embeddings_model}")
        logger.info("Векторное хранилище успешно загружено.")
        return vector_store
    except Exception as e:
        logger.error(f"Не удалось загрузить векторное хранилище: {e}")
        return None

# Основная логика
vector_store = load_vector_store()

if vector_store is None:
    logger.info("Векторное хранилище не найдено. Загрузите его самостоятельно")
