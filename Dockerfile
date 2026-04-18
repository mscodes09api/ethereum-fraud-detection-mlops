FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# THE FIX: Tell the system where the 'hidden' tools are
ENV PATH="/root/.local/bin:$PATH"

# Copy the rest of the code
COPY . .

# Expose the port Render expects
EXPOSE 10000

# Start the app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "10000"]