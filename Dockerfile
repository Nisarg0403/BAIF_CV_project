# Stage 1: Build Frontend
FROM node:20-alpine AS frontend-builder
WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm install
COPY frontend/ ./
RUN npm run build

# Stage 2: Python Backend & Serving Frontend
FROM python:3.11-slim
WORKDIR /app

# Install system dependencies for OpenCV & OS libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Create runtime data directories
RUN mkdir -p data/processed/uploaded_images

# Copy Python codebase, ML models, and configs
COPY backend/ ./backend/
COPY src/ ./src/
COPY models/ ./models/
COPY configs/ ./configs/

# Copy compiled static frontend assets from Stage 1
COPY --from=frontend-builder /app/frontend/dist ./frontend/dist

ENV PORT=8000
EXPOSE 8000

CMD ["python", "backend/main.py"]
