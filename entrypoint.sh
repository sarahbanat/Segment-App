#!/bin/bash --login

set -euo pipefail
conda activate segformer_env
exec python -m app.main