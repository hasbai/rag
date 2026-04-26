FROM python:3.12-slim
RUN pip3 install fastapi uvicorn httpx numpy
COPY proxy.py .
CMD ["uvicorn", "proxy:app", "--host", "0.0.0.0", "--port", "8080"]
