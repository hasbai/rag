FROM python:3.12-slim
RUN pip3 install fastapi uvicorn httpx
COPY embed_proxy.py .
CMD ["uvicorn", "embed_proxy:app", "--host", "0.0.0.0", "--port", "8080"]
