# sft-pipeline

Model-agnostic supervised fine-tuning. Swap models and datasets by editing `.env` only.

## 1. Deploy a pod

RunPod, A100 80GB PCIe, community Axolotl template. Container disk 50GB or more.

Axolotl comes preinstalled in a venv, so no Docker is needed. The venv uses `uv`, not `pip`.

## 2. Install Mamba kernels

Required for Nemotron and any other hybrid Mamba model. Skip for standard transformers.

```
uv pip install causal-conv1d --no-build-isolation
```

Without this the model falls back to a naive implementation: roughly 50x slower, and the loss values are garbage. It warns rather than erroring, so nothing tells you something is wrong.

## 3. Clone and configure

```
cd /workspace
git clone https://github.com/Jacob-Brokloff/sft-pipeline
cd sft-pipeline
cp .env.example .env
nano .env
```

Fill in `HF_TOKEN` and `HUB_MODEL_ID` at minimum.

## 4. Run

```
chmod +x bootstrap.sh
./bootstrap.sh
```

The adapter pushes to Hugging Face automatically when training finishes. Nothing else to run.

Not on RunPod:

```
docker run --gpus all --rm -v $(pwd):/w -w /w --env-file .env axolotlai/axolotl:main-latest ./bootstrap.sh
```

## .env reference

| Variable | What it is |
|---|---|
| `HF_TOKEN` | Fine-grained token, write scope |
| `BASE_MODEL` | Any Hugging Face model id |
| `DATASET_PATH` | Hub dataset id, or a local `.jsonl` path |
| `SPLIT` | Split name. Supports slicing, e.g. `train[:5000]` |
| `HUB_MODEL_ID` | Where the adapter gets pushed |
| `FIELD_MESSAGES` | Column holding the conversation. `messages` or `conversations` |
| `ROLE_KEY` | Role field. `role` or `from` |
| `CONTENT_KEY` | Content field. `content` or `value` |
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

OpenAI style, the default:

```
{"messages":[{"role":"user","content":"..."},{"role":"assistant","content":"..."}]}
```

ShareGPT style, set `FIELD_MESSAGES=conversations`, `ROLE_KEY=from`, `CONTENT_KEY=value`:

```
{"conversations":[{"from":"human","value":"..."},{"from":"gpt","value":"..."}]}
```

Flat tabular datasets are not supported. Convert them to one of the above first.

## Swapping models

Change `BASE_MODEL` and `HUB_MODEL_ID`. Nothing else is model specific.

`lora_target_linear: true` targets every linear layer, so it adapts to whatever architecture you point it at. `chat_template: tokenizer_default` picks up whatever template ships with the model.

Worth checking when you swap:

- `TRUST_REMOTE_CODE` off for standard Llama and Qwen, on for custom architectures
- `ADAPTER=qlora` for anything too large to fit in bf16 on your GPU
- `SEQUENCE_LEN` against the model's real context limit
