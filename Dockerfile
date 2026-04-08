FROM python:3.12-slim

WORKDIR /app

RUN pip install uv

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project

COPY . .
RUN uv sync --frozen

EXPOSE 7860
CMD ["sh", "-c", "uv run server --host 0.0.0.0 --port ${PORT:-7860}"]
