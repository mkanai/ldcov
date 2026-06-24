# ldcov Dockerfile
# ldcov is pure Python; its BGEN reader (lazybgen) installs from a prebuilt PyPI
# wheel, so no C/C++ toolchain is needed at install time.
FROM python:3.11-slim

# Runtime utilities: wget/tar to fetch bgenix, OpenBLAS for optimized linear
# algebra.
RUN apt-get update && apt-get install -y \
    wget \
    tar \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

# Install bgenix from Oxford (used to create .bgi indices for BGEN files)
RUN wget -q https://www.chg.ox.ac.uk/~gav/resources/bgen_v1.1.4-Ubuntu16.04-x86_64.tgz -O /tmp/bgen.tgz && \
    tar -xzf /tmp/bgen.tgz -C /tmp && \
    cp /tmp/bgen_v1.1.4-Ubuntu16.04-x86_64/bgenix /usr/local/bin/ && \
    chmod +x /usr/local/bin/bgenix && \
    rm -rf /tmp/bgen.tgz /tmp/bgen_v1.1.4-Ubuntu16.04-x86_64

WORKDIR /app
COPY . .

RUN pip install --upgrade pip setuptools wheel

# Install ldcov (pulls the lazybgen wheel from PyPI) with dev extras
RUN pip install -e .[dev] -v 2>&1 | tee /tmp/build.log && \
    python -c "import ldcov; print('ldcov installed at:', ldcov.__file__)" && \
    python -c "import lazybgen; print('lazybgen', lazybgen.__version__)"

# Verify numpy is using OpenBLAS
RUN python -c "import numpy; numpy.show_config()" | grep -i blas || true

# Run tests to verify the build (skip integration tests that need cloud access)
RUN pytest -v -m "not integration" tests/ || true

ENV PYTHONUNBUFFERED=1
ENV OMP_NUM_THREADS=4
ENV MKL_NUM_THREADS=4
ENV NUMEXPR_NUM_THREADS=4
ENV OPENBLAS_NUM_THREADS=4

# Create directory for test data
RUN mkdir -p /data

# Default command
CMD ["pytest", "-v"]
