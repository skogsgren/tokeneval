#!/usr/bin/env bash
## USAGE: run as current user, should be in docker group
set -euo pipefail

IMAGE_NAME="cuda12"
CONTAINER_NAME="typhoon_debug"
WORKSPACE="/hd2/gugr7935/typhoon/"
SEED_TEXT="Let's play a game of identify the animals! The dog is an animal. The cat is an animal. The train is not an animal. The rhino is"

echo "$(date) debugging generation in container $CONTAINER_NAME..."
docker run --rm --gpus all \
  --name $CONTAINER_NAME \
  -e "SEED_TEXT=$SEED_TEXT" \
  --user "$(id -u)":"$(id -g)" \
  -v $WORKSPACE:/workspace/\
  $IMAGE_NAME \
  bash -c "
    set -euo pipefail
    export HOME=/workspace
    export PATH=/workspace/.local/bin:$PATH
    cd /workspace/

    python3 generate.py --model output/tur_Latn-eng_Latn-omp/ --seed_text \"$SEED_TEXT\"
    python3 generate.py --model output/arb_Arab-eng_Latn-omp/ --seed_text \"$SEED_TEXT\"
    python3 generate.py --model output/ell_Grek-eng_Latn-omp/ --seed_text \"$SEED_TEXT\"
    python3 generate.py --model output/est_Latn-eng_Latn-omp/ --seed_text \"$SEED_TEXT\"
    python3 generate.py --model output_finetuned_mono/tur_Latn-eng_Latn-omp_finetuned/ --seed_text \"$SEED_TEXT\"
    python3 generate.py --model output_finetuned_mono/est_Latn-eng_Latn-omp_finetuned/ --seed_text \"$SEED_TEXT\"
    python3 generate.py --model output_finetuned_mono/ell_Grek-eng_Latn-omp_finetuned/ --seed_text \"$SEED_TEXT\"
    python3 generate.py --model output_finetuned_mono/arb_Arab-eng_Latn-omp_finetuned/ --seed_text \"$SEED_TEXT\"
    python3 generate.py --model goldfish-models/ell_grek_100mb --seed_text \"$SEED_TEXT\"
    python3 generate.py --model goldfish-models/arb_arab_100mb --seed_text \"$SEED_TEXT\"
    python3 generate.py --model goldfish-models/eng_Latn_100mb --seed_text \"$SEED_TEXT\"
  "
echo "$(date) finished debugging"
