FROM python:3.12-slim

# Install uv.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set the working directory.
WORKDIR /app

# Install the application dependencies.
COPY uv.lock pyproject.toml README.md ./
RUN uv sync --frozen --no-cache

# Copy the application into the container.
COPY src/RAG_Agent RAG_Agent/

CMD ["/app/.venv/bin/fastapi", "run", "RAG_Agent/infrastructure/api/main.py", "--port", "8000", "--host", "0.0.0.0"]

