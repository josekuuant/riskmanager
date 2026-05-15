FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY api.py ./

# Railway/Render setean $PORT; Fly.io usa 8080; default 8000 para local.
ENV PORT=8000
EXPOSE 8000

# Shell form para que $PORT se expanda
CMD uvicorn api:app --host 0.0.0.0 --port ${PORT}
