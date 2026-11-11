FROM python:3.11-slim-bookworm AS dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl unzip libgomp1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /work
COPY requirements.txt requirements-torch.txt ./
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
RUN python -m pip install --no-cache-dir pip==24.0 setuptools==65.5.0 wheel==0.38.4 \
    && python -m pip install --no-cache-dir -r requirements.txt \
    && python -m pip install --no-cache-dir --no-build-isolation gym==0.21.0 \
    && python -m pip install --no-cache-dir --no-deps robomimic==0.2.0

FROM dependencies AS cpu
RUN python -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu -r requirements-torch.txt
COPY . .

FROM dependencies AS cuda
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
RUN python -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu121 -r requirements-torch.txt
COPY . .
