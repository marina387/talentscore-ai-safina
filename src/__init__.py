"""TalentScore AI Safina — RAG-шаблон из коробки.

Пакет содержит MVP реализацию RAG (Retrieval-Augmented Generation):
- чанкинг текста с перекрытием (overlap)
- простой retrieval на базе TF-IDF + cosine similarity
- pipeline для построения индекса и поиска по вопросам
- загрузка документов (txt/pdf)
- CLI демо

Основной API:
    from src import RagPipeline, Chunk, TfidfRetriever, load_documents_from_folder

    # Построить индекс
    rag = RagPipeline()
    rag.build_index(documents)

    # Задать вопрос
    result = rag.ask("Нужен ли фундамент?", top_k=5)
    print(result.as_text())

На следующих этапах проекта можно заменить TF-IDF на sentence-transformers
и подключить реальную LLM для генерации ответов.
"""

__version__ = "0.1.0"

from src.core.chunking import Chunk
from src.core.rag_pipeline import RagPipeline, RagResult
from src.core.retrieval import SearchHit, TfidfRetriever
from src.parsers.loaders import load_documents_from_folder

__all__ = [
    "RagPipeline",
    "RagResult",
    "Chunk",
    "TfidfRetriever",
    "SearchHit",
    "load_documents_from_folder",
    "__version__",
]
