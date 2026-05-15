FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY api.py ./

# Railway sets $PORT dynamically (usually 8080). For local dev, default to 8080
# so EXPOSE / health checks / Railway target_port all line up out of the box.
ENV PORT=8080
EXPOSE 8080

# JSON form via sh -c so ${PORT} still expands at runtime.
CMD ["sh", "-c", "uvicorn api:app --host 0.0.0.0 --port ${PORT}"]
