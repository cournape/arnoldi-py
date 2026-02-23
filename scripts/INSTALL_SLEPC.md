# Installing SLEPc (slepc4py) in the project venv

This project uses a Python 3.11 virtual environment managed by `uv`. The
`slepc4py` (and `petsc4py`) Python bindings must be compiled from source
against the system PETSc/SLEPc shared libraries because:

- No pre-built wheels exist for the aarch64 platform.
- The system libraries expose the correct ABI for linking.
- The `uv`-managed venv isolates system site-packages by default.

Note: those instructions were AI-generated after letting claude code figure out
how to install SLPEc on debian 13 arm64

## Prerequisites

### System packages (Debian/Ubuntu)

Install the PETSc and SLEPc development packages (libraries + headers),
as well as MPI:

```sh
sudo apt install \
    libpetsc-real-dev \
    libslepc-real-dev \
    libopenmpi-dev
```

The packages above pull in:
- `libpetsc-real3.22` / `libpetsc-real3.22-dev` (PETSc 3.22.5)
- `libslepc-real3.22` / `libslepc-real3.22-dev` (SLEPc 3.22.2)
- `mpicc` / `mpiexec` wrappers

After installation the system provides two canonical symlinks:

| Symlink | Points to |
|---------|-----------|
| `/usr/lib/petsc` | `/usr/lib/petscdir/petsc3.22/aarch64-linux-gnu-real` |
| `/usr/lib/slepc` | `/usr/lib/slepcdir/slepc3.22/aarch64-linux-gnu-real` |

These are the values used for `PETSC_DIR` / `SLEPC_DIR` below.

## Setting up the venv

Create or sync the project venv with `uv` (Python 3.11 is required):

```sh
uv sync
```

## Installing petsc4py and slepc4py

### 1. Install a compatible Cython version

`petsc4py` 3.22.x ships a custom Cython autodoc helper (`cyautodoc.py`)
that uses internal Cython APIs. Cython ≥ 3.1 changed those APIs and causes
a compiler crash. Pin Cython to the 3.0.x series:

```sh
uv pip install "cython>=3.0,<3.1"
```

### 2. Build and install petsc4py

Point the build at the system PETSc installation and use
`--no-build-isolation` so the pinned Cython from the venv is used:

```sh
PETSC_DIR=/usr/lib/petsc SLEPC_DIR=/usr/lib/slepc \
  uv pip install --no-build-isolation petsc4py==3.22.4
```

The compilation takes roughly 2–3 minutes on aarch64.

### 3. Build and install slepc4py

```sh
PETSC_DIR=/usr/lib/petsc SLEPC_DIR=/usr/lib/slepc \
  uv pip install --no-build-isolation slepc4py==3.22.2
```

## Running Python with slepc4py

The environment variables `PETSC_DIR` and `SLEPC_DIR` must be set at
**runtime** as well so that the shared libraries are found:

```sh
PETSC_DIR=/usr/lib/petsc SLEPC_DIR=/usr/lib/slepc python <your_script.py>
```

Quick sanity check:

```sh
PETSC_DIR=/usr/lib/petsc SLEPC_DIR=/usr/lib/slepc .venv/bin/python - <<'EOF'
import petsc4py, slepc4py
petsc4py.init()
slepc4py.init()
from slepc4py import SLEPc
eps = SLEPc.EPS().create()
print("petsc4py:", petsc4py.__version__)
print("slepc4py:", slepc4py.__version__)
print("SLEPc EPS created OK")
eps.destroy()
EOF
```

Expected output:

```
petsc4py: 3.22.4
slepc4py: 3.22.2
SLEPc EPS created OK
```

## Version matrix

| Component | Version |
|-----------|---------|
| Python    | 3.11.13 |
| uv        | 0.10.4  |
| PETSc     | 3.22.5  |
| SLEPc     | 3.22.2  |
| petsc4py  | 3.22.4  |
| slepc4py  | 3.22.2  |
| Cython    | 3.0.12  |
