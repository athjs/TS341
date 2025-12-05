FROM python:3.13-slim

# Installation des dépendances système pour OpenCV
RUN apt-get update && \
    apt-get install --no-install-recommends -y \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    libsm6 \
    libxext6 \
    libxrender-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app/


#WARN: Obligé de passer par pip, poetry ne prennait pas les librairies en compte sinon.
RUN pip install --no-cache-dir \
    "numpy>=2,<2.3.0" \
    "opencv-python>=4" \
    "flask>=3.1.2" \
    "ultralytics>=8.3.228,<9.0.0" \
    "opencv-contrib-python>=4.12.0.88,<5.0.0.0"

COPY ./ts341_project/ /app/ts341_project

RUN mkdir -p /app/videos
ENTRYPOINT ["python", "ts341_project/filtre/filtre.py"]
CMD ["video2_short"]
