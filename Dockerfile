FROM python:3.8
WORKDIR /home/hanna/Documents/Meta-optimization/L2L
COPY . .
RUN python -m pip install -e .
LABEL org.opencontainers.image.source="https://github.com/Meta-optimization/L2L"
#CMD ["python", "./bin/l2l-fun-ga.py"]