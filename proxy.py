from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import httpx
import os

app = FastAPI()
UPSTREAM = os.environ.get("UPSTREAM", "http://llama:8080")
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
    return JSONResponse(content=resp.json(), status_code=resp.status_code)

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
