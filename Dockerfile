FROM python:3.12-slim

# Install uv.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set the working directory.
WORKDIR /app

# Install the application dependencies.
COPY uv.lock pyproject.toml ./
RUN uv sync --frozen --no-cache

# Pre-download the fastembed BM25 model at build time so the runtime
# container never hits HuggingFace. Cloud Run's egress IPs are shared
# across many GCP customers; HuggingFace's anonymous rate-limit (~500
# req/5 min/IP) fires on the first cold start, every cold start retries,
# and the container fails to bind in time → 503. Baking the model into
# the image layer is the simplest, most production-ready fix.
RUN uv run python -c "from fastembed import SparseTextEmbedding; SparseTextEmbedding('Qdrant/bm25')"


# Copy the application into the container.
COPY src/RAG_Agent RAG_Agent/

CMD ["/app/.venv/bin/fastapi", "run", "RAG_Agent/infrastructure/api/main.py", "--port", "8080", "--host", "0.0.0.0"]

