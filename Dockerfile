# --- Stage 1: Build ---
FROM python:3.11-slim as builder

WORKDIR /app

COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# --- Stage 2: Final Image ---
FROM python:3.11-slim

# Create a non-root user for security
RUN groupadd -r appgroup && useradd -r -g appgroup appuser

WORKDIR /app

# Copy built dependencies from builder
COPY --from=builder /root/.local /home/appuser/.local
COPY . .

# Ensure scripts are in PATH and set permissions
ENV PATH=/home/appuser/.local/bin:$PATH
RUN chown -R appuser:appgroup /app

USER appuser

# Expose port and run application
EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
