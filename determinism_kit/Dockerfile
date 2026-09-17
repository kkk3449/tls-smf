# TLS-SMF Stage A (deterministic geometric backbone) — cross-machine
# determinism verification image.
#
# Build (from the determinism_kit/ directory):
#   docker build -t tlssmf-determinism .
# Run (mount a directory that contains the .e57 scan):
#   docker run --rm -v /path/to/scan_dir:/data tlssmf-determinism "/data/vis_n2 1.e57"
# The report is written to /data/determinism_report.json.
# Pinned by digest (= python 3.12.14-slim, Debian trixie, glibc 2.41) so a
# rebuild on another machine resolves to the identical base image.
FROM python:3.12-slim@sha256:09f7da3bc104798d0afb40bc08d23ab2da20a76130cec1f2ef170848f5d85217

# open3d needs libGL / libgomp even for headless CPU use
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 libgomp1 libegl1 libusb-1.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/tlssmf
COPY requirements-pinned.txt .
RUN pip install --no-cache-dir -r requirements-pinned.txt

# Backbone code only (blk360seg/, scripts/, configs/ — no outputs, no weights)
COPY code/blk360seg ./blk360seg
COPY code/scripts   ./scripts
COPY code/configs   ./configs
COPY verify_determinism.sh make_report.py ./
RUN chmod +x verify_determinism.sh

ENTRYPOINT ["/opt/tlssmf/verify_determinism.sh"]
