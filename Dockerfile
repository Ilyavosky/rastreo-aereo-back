FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /srv
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY app ./app

ENV PATH="/srv/.venv/bin:$PATH"
ENV PORT=8000
EXPOSE ${PORT}

CMD ["sh", "-c", "fastapi run app/main.py --host 0.0.0.0 --port ${PORT}"]
