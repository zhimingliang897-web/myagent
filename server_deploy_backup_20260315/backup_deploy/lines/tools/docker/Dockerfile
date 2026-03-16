FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir flask requests pdfplumber edge-tts

# 从上层目录复制源码（由 docker-compose build context 控制）
COPY pdf_extractor.py pop_mystic_scraper.py quote_formatter.py tts_generator.py app.py ./
COPY templates/ templates/

RUN mkdir -p word

VOLUME /app/word

EXPOSE 5000

CMD ["python", "app.py"]
