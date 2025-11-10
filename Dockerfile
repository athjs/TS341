FROM python:3.13-slim AS builder
RUN apt-get update && pip install --no-cache-dir poetry==1.8.4 
WORKDIR /app/
COPY ./poetry.lock ./pyproject.toml /app/
RUN poetry install 
COPY ./ts341_project/ app/
RUN apt-get update && \
    apt-get install --no-install-recommends -y \
    vim \
    python3 \
    curl && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
COPY --from=builder /app/dist /usr/share/
