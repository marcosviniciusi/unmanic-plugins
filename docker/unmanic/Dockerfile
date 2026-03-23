FROM josh5/unmanic:latest

LABEL maintainer="marcosviniciusi"
LABEL description="Unmanic with OCR support for PGS/VOBSUB subtitle conversion to SRT"

# Install tesseract-ocr, language packs, and mkvtoolnix
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-por \
    tesseract-ocr-eng \
    mkvtoolnix \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install pgsrip into the Unmanic virtualenv (/opt/venv) and system Python
RUN /opt/venv/bin/pip install --no-cache-dir pgsrip && \
    pip3 install --no-cache-dir --break-system-packages pgsrip

# Download tessdata_best for Portuguese (better accuracy on accented characters)
RUN TESSDATA_DIR=$(find / -type d -name tessdata 2>/dev/null | head -1) && \
    if [ -n "$TESSDATA_DIR" ]; then \
        wget -q -O "${TESSDATA_DIR}/por.traineddata" \
            "https://github.com/tesseract-ocr/tessdata_best/raw/main/por.traineddata" && \
        wget -q -O "${TESSDATA_DIR}/eng.traineddata" \
            "https://github.com/tesseract-ocr/tessdata_best/raw/main/eng.traineddata"; \
    fi

# Verify installations (both system and venv)
RUN tesseract --version && \
    /opt/venv/bin/python3 -c "import pgsrip; print('pgsrip OK (venv)')" && \
    python3 -c "import pgsrip; print('pgsrip OK (system)')" && \
    mkvextract --version
