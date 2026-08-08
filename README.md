# sft-pipeline

Model-agnostic supervised fine-tuning. Swap models by editing `.env` only.

```
docker run --gpus all --rm -v $(pwd):/w -w /w --env-file .env axolotlai/axolotl:main-latest ./bootstrap.sh
```

## On a fresh pod

```
git clone https://github.com/Jacob-Brokloff/sft-pipeline
cd sft-pipeline
cp .env.example .env
nano .env
chmod +x bootstrap.sh
```

Then run the command at the top.

## .env

| Variable | What it is |
|---|---|
| `HF_TOKEN` | Fine-grained token, write scope |
| `BASE_MODEL` | Any Hugging Face model id |
| `DATASET_PATH` | Hub dataset id, or a local `.jsonl` path |
| `HUB_MODEL_ID` | Where the adapter gets pushed |
| `NUM_EPOCHS` | Default 2 |
| `VAL_SET_SIZE` | Default 0.05 |
| `SEQUENCE_LEN` | Default 2048 |
| `LEARNING_RATE` | Default 0.0002 |
| `ADAPTER` | `lora` or `qlora` |
| `LORA_R` / `LORA_ALPHA` | Default 32 / 64 |
| `TRUST_REMOTE_CODE` | Needed for custom architectures |
| `SAMPLE_PACKING` | Off by default. Faster when on, not safe on every architecture |
| `FLASH_ATTENTION` | Off by default |
| `MERGE` | `true` to also merge the adapter into base weights |

## Dataset format

One JSON object per line:

```
{"messages":[{"role":"user","content":"..."},{"role":"assistant","content":"..."}]}
```

## Swapping models

Change `BASE_MODEL` and `HUB_MODEL_ID`. Nothing else is model specific.

`lora_target_linear: true` targets every linear layer, so it adapts to whatever architecture you point it at. `chat_template: tokenizer_default` picks up whatever template ships with the model.

Per-model flags worth checking when you swap:

- `TRUST_REMOTE_CODE` off for standard Llama and Qwen, on for custom architectures
- `ADAPTER=qlora` for anything too large to fit in bf16 on your GPU
- `SEQUENCE_LEN` against the model's real context limit

## Notes

The adapter pushes to Hugging Face automatically when training finishes. Nothing else to run.
# sft-pipeline
