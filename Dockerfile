FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY api.py ./

# Railway sets $PORT at runtime (usually 8080). The app must listen on
# that exact value — Railway's healthcheck and public routing both use it.
EXPOSE 8080
CMD ["sh", "-c", "uvicorn api:app --host 0.0.0.0 --port ${PORT:-8080}"]
