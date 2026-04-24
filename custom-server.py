import os

from fastapi import APIRouter, HTTPException
from lightrag.api.routers.document_routes import create_document_routes
from lightrag.lightrag import LightRAG
from lightrag.operate import chunking_by_token_size
from lightrag.utils import logger
from pydantic import BaseModel, Field

CUSTOM_CHUNK_SEPARATOR = os.getenv("CUSTOM_CHUNK_SEPARATOR", "=====CHUNK=====")


def custom_chunking_func(
    tokenizer,
    content,
    split_by_character,  # 框架传入的值会被忽略
    split_by_character_only,
    chunk_overlap_token_size,
    chunk_token_size,
):

    return chunking_by_token_size(
        tokenizer=tokenizer,
        content=content,
        split_by_character=CUSTOM_CHUNK_SEPARATOR,  # 强制指定
        split_by_character_only=split_by_character_only,
        chunk_overlap_token_size=chunk_overlap_token_size,
        chunk_token_size=chunk_token_size,
    )


def create_pre_chunked_routes(rag: LightRAG):

    router = APIRouter(tags=["documents"])

    class PreChunkedRequest(BaseModel):
        chunks: list[dict[str, str]] = Field(
            ...,
            description="List of chunk dicts, each with 'content' (required) "
            "Optional keys: 'file_path', 'chunk_order_index'.",
        )
        file_path: str = Field(
            default="pre-chunked",
            description="Default file_path for chunks that don't specify one.",
        )

        model_config = {
            "json_schema_extra": {
                "example": {
                    "chunks": [
                        {
                            "content": "First chunk content",
                        },
                        {
                            "content": "Second chunk content",
                        },
                    ],
                    "file_path": "my_data.txt",
                }
            }
        }

    class PreChunkedResponse(BaseModel):
        status: str
        message: str
        chunk_count: int

    @router.post("/pre-chunked", response_model=PreChunkedResponse)
    async def insert_pre_chunked(request: PreChunkedRequest):
        """Insert pre-chunked documents directly, bypassing the chunking pipeline.

        Each chunk must have 'content' and 'source_id'. The chunking step is
        completely skipped — chunks are inserted directly into the knowledge graph
        pipeline (entity extraction, embedding, etc.).
        """

        for i, chunk in enumerate(request.chunks):
            chunk["file_path"] = request.file_path
            chunk["source_id"] = request.file_path
            if "chunk_order_index" not in chunk:
                chunk["chunk_order_index"] = i

        try:
            await rag.ainsert_custom_kg({"chunks": request.chunks})
            logger.info(
                "Pre-chunked insertion: %d chunks inserted (file_path=%s)",
                len(request.chunks),
                request.file_path,
            )
            return PreChunkedResponse(
                status="success",
                message=f"Successfully inserted {len(request.chunks)} pre-chunked documents.",
                chunk_count=len(request.chunks),
            )
        except Exception as e:
            logger.error(f"Error inserting pre-chunked data: {e}")
            logger.exception(e)
            raise HTTPException(status_code=500, detail=str(e))

    return router


def patched_create_doc_routes(rag, doc_manager, api_key=None):
    router = create_document_routes(rag, doc_manager, api_key)
    pre_chunked_routes = create_pre_chunked_routes(rag)
    router.include_router(pre_chunked_routes)
    logger.info("Added /documents/pre-chunked endpoint for pre-chunked JSON upload")
    return router


if __name__ == "__main__":
    LightRAG.__dataclass_fields__["chunking_func"].default_factory = lambda: (
        custom_chunking_func
    )

    from lightrag.api import lightrag_server as _server_mod

    _server_mod.create_document_routes = patched_create_doc_routes

    from lightrag.api.lightrag_server import main

    main()
