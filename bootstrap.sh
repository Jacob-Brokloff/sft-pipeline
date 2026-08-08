#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

if [ -f .env ]; then
  set -a; . ./.env; set +a
fi

: "${HF_TOKEN:?missing HF_TOKEN}"
: "${BASE_MODEL:?missing BASE_MODEL}"
: "${DATASET_PATH:?missing DATASET_PATH}"
: "${HUB_MODEL_ID:?missing HUB_MODEL_ID}"
: "${NUM_EPOCHS:=2}"
: "${VAL_SET_SIZE:=0.05}"
: "${SEQUENCE_LEN:=2048}"
: "${LEARNING_RATE:=0.0002}"
: "${MERGE:=false}"
: "${ADAPTER:=lora}"
: "${LORA_R:=32}"
: "${LORA_ALPHA:=64}"
: "${TRUST_REMOTE_CODE:=true}"
: "${SAMPLE_PACKING:=false}"
: "${FLASH_ATTENTION:=false}"
: "${FIELD_MESSAGES:=messages}"
: "${ROLE_KEY:=role}"
: "${CONTENT_KEY:=content}"

case "$DATASET_PATH" in
  *.jsonl|*.json) DATASET_DS_TYPE="ds_type: json" ;;
  *)              DATASET_DS_TYPE="" ;;
esac

export BASE_MODEL DATASET_PATH DATASET_DS_TYPE HUB_MODEL_ID \
       NUM_EPOCHS VAL_SET_SIZE SEQUENCE_LEN LEARNING_RATE \
       ADAPTER LORA_R LORA_ALPHA TRUST_REMOTE_CODE SAMPLE_PACKING FLASH_ATTENTION \
       FIELD_MESSAGES ROLE_KEY CONTENT_KEY
export HF_TOKEN HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"

command -v envsubst >/dev/null 2>&1 || {
  apt-get update -qq && apt-get install -y -qq gettext-base
}

if command -v hf >/dev/null 2>&1; then
  hf auth login --token "$HF_TOKEN"
else
  huggingface-cli login --token "$HF_TOKEN"
fi

envsubst < config.yml.template | sed '/^[[:space:]]*$/d' > config.yml
cat config.yml

axolotl train config.yml

if [ "$MERGE" = "true" ]; then
  axolotl merge-lora config.yml --lora-model-dir=./out
fi

echo "https://huggingface.co/${HUB_MODEL_ID}"
