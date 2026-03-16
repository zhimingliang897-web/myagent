FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY config.py llm_client.py debate_engine.py tts_engine.py video_export.py topics.py app.py ./
COPY templates/ templates/
COPY static/ static/

VOLUME /app/output

EXPOSE 5000

CMD ["python", "app.py"]
