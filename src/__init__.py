"""Source package for DocAnalyzer AI Pro."""

from .config import (
    APP_TITLE,
    BASE_DIR,
    DEFAULT_MODEL,
    EMBEDDING_MODEL,
    MODEL_OPTIONS,
    SUPPORTED_DOCUMENT_TYPES,
    SUPPORTED_IMAGE_TYPES,
    SUPPORTED_UPLOAD_TYPES,
    UPLOADS_DIR,
)
from .dataset import (
    add_text_to_nodes,
    build_tree,
    extract_markdown_nodes,
    extract_pdf_text_and_images,
    extract_txt_nodes,
    flatten_tree,
    parse_document,
    parse_markdown,
)
from .inference import ask_gemini, ask_gemini_vision, build_prompt, clean_answer
from .preprocessing import split_text
from .rag import (
    build_context,
    cosine_similarity,
    get_embeddings_for_nodes,
    keyword_score,
    normalize_text,
    select_relevant_nodes,
    select_relevant_nodes_semantic,
)
from .utils import get_file_hash, get_mime_type, safe_filename, save_uploaded_file

__all__ = [
    "APP_TITLE",
    "BASE_DIR",
    "UPLOADS_DIR",
    "SUPPORTED_DOCUMENT_TYPES",
    "SUPPORTED_IMAGE_TYPES",
    "SUPPORTED_UPLOAD_TYPES",
    "MODEL_OPTIONS",
    "DEFAULT_MODEL",
    "EMBEDDING_MODEL",
    "split_text",
    "extract_pdf_text_and_images",
    "extract_txt_nodes",
    "extract_markdown_nodes",
    "add_text_to_nodes",
    "build_tree",
    "flatten_tree",
    "parse_markdown",
    "parse_document",
    "normalize_text",
    "keyword_score",
    "select_relevant_nodes",
    "cosine_similarity",
    "get_embeddings_for_nodes",
    "select_relevant_nodes_semantic",
    "build_context",
    "ask_gemini_vision",
    "clean_answer",
    "build_prompt",
    "ask_gemini",
    "get_file_hash",
    "safe_filename",
    "save_uploaded_file",
    "get_mime_type",
]
