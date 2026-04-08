FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Expose port if needed
EXPOSE 7860

# Default command
CMD ["python", "baseline_inference.py"]