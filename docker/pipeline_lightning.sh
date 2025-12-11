#!/usr/bin/env bash
## USAGE: run as current user, should be in docker group
set -e

IMAGE_NAME="cuda12"
CONTAINER_NAME="typhoon"
WORKSPACE="/hd2/gugr7935/typhoon/"

echo "$(date) starting evaluation in container $CONTAINER_NAME..."
docker run --rm --gpus all \
  --name $CONTAINER_NAME \
  --user "$(id -u)":"$(id -g)" \
  -v $WORKSPACE:/workspace/\
  $IMAGE_NAME \
  bash -c "
    set -euo pipefail
    export HOME=/workspace &&
    export PATH=/workspace/.local/bin:$PATH
    cd /workspace/ &&
    echo '$(date) installing dependencies...' &&
    pip install -U pip &&
    pip install --no-cache-dir -r ./requirements.txt &&
    pip install torch==2.6.0 &&
    pip install -e mergekit-git/ &&
    echo '$(date) running pipeline. this will take a while. grab a coffee...' &&
    python3 pipeline.py cfg_lightning.yaml --no_transfer
  "
echo "$(date) finished pipeline"
