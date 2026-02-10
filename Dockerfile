FROM python:3.9-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DATABASE_URL=sqlite:////app/data/cv_app.db

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r /app/requirements.txt

COPY app /app/app
COPY .env.example /app/.env.example

# Runtime directories for persistent DB + generated files.
RUN mkdir -p /app/data /app/files/original /app/files/tailored

EXPOSE 8000

VOLUME ["/app/data", "/app/files"]

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
