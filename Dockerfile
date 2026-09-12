# Multi-stage image: build with uv, run as non-root (tech.md 16.3).
FROM python:3.11-slim AS builder

WORKDIR /build
RUN pip install --no-cache-dir uv
COPY pyproject.toml README.md ./
COPY src ./src
RUN uv pip install --python /usr/local/bin/python --no-cache .

FROM python:3.11-slim AS runtime

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 --shell /usr/sbin/nologin app

WORKDIR /app
COPY --from=builder /usr/local /usr/local
COPY configs ./configs

RUN mkdir -p /app/data /app/runs /app/logs \
    && chown -R app:app /app

USER app
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TZ=UTC

HEALTHCHECK --interval=30s --timeout=8s --start-period=45s --retries=3 \
    CMD python -c "import tradingbot" || exit 1

ENTRYPOINT ["tradingbot"]
CMD ["live", "run", "--mode", "paper"]
