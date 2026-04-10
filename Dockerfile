FROM python:3.12-slim

WORKDIR /app

# Install build tools needed for some packages (qiskit-aer, scipy)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ && \
    rm -rf /var/lib/apt/lists/*

# Copy pinned dependencies first for layer caching.
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the full project and install it into system Python.
COPY . .
RUN pip install --no-cache-dir --no-deps .

EXPOSE 7860

CMD ["sh", "-c", "python -m uvicorn server.app:app --host 0.0.0.0 --port ${PORT:-7860}"]
