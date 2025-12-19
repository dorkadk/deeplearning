# <-- 1) Base image
FROM python:3.10-slim

# <-- 2) Environment setup
ENV PYTHONUNBUFFERED=1
WORKDIR /app

# <-- 3) System deps (ha szükséges pl. Git, build tools)
RUN apt-get update && apt-get install -y dos2unix\
    git \
 && rm -rf /var/lib/apt/lists/*

# <-- 4) Copy requirements és telepítés
COPY requirements.txt .
RUN pip install --upgrade pip \
 && pip install -r requirements.txt


COPY src ./src
COPY run.sh ./run.sh
RUN dos2unix /app/run.sh && chmod +x / app/run.sh

# <-- 6) Default parancs
# Indítsa el a fő Python scriptet
CMD ["sh", "run.sh"]
