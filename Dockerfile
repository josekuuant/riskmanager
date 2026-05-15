FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY api.py ./

# Railway's UI has Target Port hardcoded to 8000 for this service's public
# domain (see network logs: external traffic routed to :8000, healthcheck
# to :8080). Hardcode the listen port to 8000 so external traffic actually
# reaches uvicorn. If you later change the Target Port in Railway UI to
# 8080 / blank for auto-detect, this can go back to `${PORT}`.
EXPOSE 8000
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
