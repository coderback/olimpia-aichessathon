# Olimpia move API. The build context is demo/api after prepare.sh has copied agent.py and
# LICENSE into it, so game logs, zips and everything else in the repo stay out of the image.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    NUMBA_NUM_THREADS=1 \
    OMP_NUM_THREADS=1 \
    MALLOC_ARENA_MAX=2

WORKDIR /app

# The engine's own imports. torch and onnxruntime are in pyproject.toml but agent.py does not
# use them. Versions match pyproject.toml.
RUN pip install --no-cache-dir chess==1.11.2 numpy==2.5.2 numba==0.67.0

# agent.py is byte-identical to the repo's. It has numba compile the search at import, so every
# cold start spends its first stretch compiling. numba's on-disk cache cannot help: a cached
# copy of the recursive negamax fails to load (LLVM "Symbol not found").
COPY agent.py server.py LICENSE /app/

RUN useradd --system --no-create-home olimpia && chown -R olimpia /app
USER olimpia

EXPOSE 8080
CMD ["python", "server.py"]
