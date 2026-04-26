import base64
import os

import httpx
import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

app = FastAPI()
UPSTREAM = os.environ.get("UPSTREAM", "http://localhost:8080")
PREFIX = os.environ.get("PREFIX", "Document: ")


@app.post("/v1/embeddings")
async def proxy_embeddings(request: Request):
    body = await request.json()

    # 注入前缀
    inp = body.get("input", [])
    if isinstance(inp, str):
        body["input"] = PREFIX + inp
    elif isinstance(inp, list):
        body["input"] = [PREFIX + t if isinstance(t, str) else t for t in inp]

    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{UPSTREAM}/v1/embeddings",
            json=body,
            timeout=120,
        )
    # 将 embedding 字段从 float list 转成 base64，jina.py 需要这个格式
    result = resp.json()
    for item in result.get("data", []):
        emb = item.get("embedding")
        if isinstance(emb, list):
            arr = np.array(emb, dtype=np.float32)
            item["embedding"] = base64.b64encode(arr.tobytes()).decode("ascii")

    return JSONResponse(content=result, status_code=resp.status_code)


# 透传其他路由（/health 等）
@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def passthrough(path: str, request: Request):
    body = await request.body()
    async with httpx.AsyncClient() as client:
        resp = await client.request(
            method=request.method,
            url=f"{UPSTREAM}/{path}",
            content=body,
            headers=dict(request.headers),
        )
    return JSONResponse(content=resp.json(), status_code=resp.status_code)
