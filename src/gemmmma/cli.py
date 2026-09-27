import argparse
import os
import json
import subprocess
import sys
import shutil
from datasets import load_dataset
from huggingface_hub import snapshot_download

def download_model(model_id, dest):
    """
    Downloads a base model from Hugging Face into the local directory.
    Using an MLX-quantized model saves massive amounts of RAM during training.
    """
    print(f"==> Downloading MLX model '{model_id}' to '{dest}'...")
    snapshot_download(repo_id=model_id, local_dir=dest)
    print("==> Download complete.")

def prep_dataset(dataset_id, dataset_config, dest_dir, max_samples=None):
    """
    Downloads a dataset from Hugging Face and prepares it into the JSONL 
    ChatML format required by MLX for fine-tuning.
    """
    print(f"==> Downloading and preparing dataset '{dataset_id}'...")
    os.makedirs(dest_dir, exist_ok=True)
    
    try:
        from datasets import Audio
        ds = load_dataset(dataset_id, name=dataset_config, split="train")
        if "audio" in ds.features:
            ds = ds.cast_column("audio", Audio(decode=False))
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("\nNOTE: Some datasets are 'gated'. To use them, you must:")
        print("  1. Go to their Hugging Face dataset page and accept the terms.")
        print("  2. Run `hf auth login` in your terminal.")
        print("For an open audio dataset test, try: --dataset PolyAI/minds14 --dataset-config en-US")
        return

    # Shuffle to ensure a good mix of examples, and slice for fast prototyping
    ds = ds.shuffle(seed=42)
    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))
    
    # Split: 90% for training the weights, 10% for validating the loss
    split_ds = ds.train_test_split(test_size=0.1, seed=42)
    
    def format_row(row):
        """
        Converts the specific columns of your dataset into a universal 
        OpenAI/ChatML 'messages' array. 
        Note: When handling multimodal Gemma 4 data, this is where you will 
        inject {"type": "image", ...} dictionaries into the content array!
        """
        # For PolyAI/minds14 dataset mapping
        if "english_transcription" in row:
            prompt = "Transcribe the following audio:"
            completion = row["english_transcription"]
        else:
            prompt = row.get("instruction", row.get("prompt", "Analyze the provided input."))
            if row.get("input"): 
                prompt += "\n\nContext: " + row.get("input", "")
            completion = row.get("output", row.get("completion", row.get("response", "")))
        
        audio_data = row.get("audio", None)
        audio_path = None
        
        # Hugging Face 'Audio' feature returns a dict with 'path'
        if isinstance(audio_data, dict) and "path" in audio_data:
            audio_path = audio_data["path"]
        elif isinstance(audio_data, str):
            audio_path = audio_data
        
        if audio_path:
            # Gemma 4 Interleaved ChatML standard
            user_content = [
                {"type": "audio", "audio": audio_path},
                {"type": "text", "text": prompt}
            ]
            assistant_content = [
                {"type": "text", "text": completion}
            ]
        else:
            user_content = prompt
            assistant_content = completion
            
        return {
            "messages": [
                {"role": "user", "content": user_content}, 
                {"role": "assistant", "content": assistant_content}
            ]
        }

    print(f"==> Formatting {len(split_ds['train'])} training examples...")
    train_data = [format_row(row) for row in split_ds['train']]
    valid_data = [format_row(row) for row in split_ds['test']]
    
    # Write exactly one JSON object per line (JSONL)
    with open(os.path.join(dest_dir, "train.jsonl"), "w") as f:
        for item in train_data:
            f.write(json.dumps(item) + "\n")
            
    with open(os.path.join(dest_dir, "valid.jsonl"), "w") as f:
        for item in valid_data:
            f.write(json.dumps(item) + "\n")
            
    print(f"==> Dataset prepped and saved to {dest_dir}/")

class MLXMonitorCallback:
    """Callback hook for streaming training & validation telemetry to training_log.jsonl."""
    def __init__(self, log_path):
        self.log_path = log_path

    def on_train_loss_report(self, train_info: dict):
        entry = {
            "type": "train",
            "iter": int(train_info.get("iteration", 0)),
            "loss": round(float(train_info.get("train_loss", 0.0)), 4),
            "learning_rate": float(train_info.get("learning_rate", 0.0)),
            "it_sec": round(float(train_info.get("iterations_per_second", 0.0)), 2),
            "tokens_sec": round(float(train_info.get("tokens_per_second", 0.0)), 1),
            "trained_tokens": int(train_info.get("trained_tokens", 0)),
            "peak_mem_gb": round(float(train_info.get("peak_memory", 0.0)), 3)
        }
        with open(self.log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def on_val_loss_report(self, val_info: dict):
        entry = {
            "type": "val",
            "iter": int(val_info.get("iteration", 0)),
            "loss": round(float(val_info.get("val_loss", 0.0)), 4),
            "val_took_s": round(float(val_info.get("val_time", 0.0)), 2)
        }
        with open(self.log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")


def _is_vlm_model(model_path, explicit_multimodal=False):
    """Detects whether training should use the Multimodal (VLM / Audio) pipeline."""
    return bool(explicit_multimodal)


def run_train(model_path, data_path, iters, batch_size, log_file="training_log.jsonl", multimodal=False, tune_audio_encoder=False, qat=False, rank=8, lora_layers=16, learning_rate=1e-5):
    """
    Executes the LoRA training loop on Apple GPU (Metal) using mlx-tune.
    Freezes base model weights and trains low-rank adapters with native telemetry.
    """
    import re
    import datetime
    import time
    from pathlib import Path
    
    print(f"==> Starting LoRA training with mlx-tune engine on Apple Silicon...")
    if qat:
        print("==> [QAT Optimization] Enforcing conservative rank and unquantized base checkpoint flow.")

    is_vlm = _is_vlm_model(model_path, explicit_multimodal=multimodal)
    print(f"==> Architecture pipeline: {'Multimodal (VLM / Audio / Vision)' if is_vlm else 'Standard Causal LM'}")

    start_time = datetime.datetime.now(datetime.timezone.utc).isoformat()
    with open(log_file, "w") as f:
        f.write(json.dumps({"type": "status", "status": "started", "timestamp": start_time}) + "\n")
        f.write(json.dumps({
            "type": "config",
            "model": model_path,
            "data": data_path,
            "total_iters": iters,
            "batch_size": batch_size,
            "multimodal": is_vlm,
            "model_style": "QAT" if qat else "Standard",
            "rank": rank,
            "lora_layers": lora_layers
        }) + "\n")

    trained_native = False

    # Try programmatic MLX-Tune training
    try:
        if is_vlm:
            from mlx_tune import FastVisionModel, UnslothVisionDataCollator
            from mlx_tune.vlm import _VLMTrainerShim, _detect_assistant_role_token
            from mlx_vlm.trainer.sft_trainer import save_adapter
            import mlx.core as mx
            import mlx.nn as nn
            import mlx.optimizers as optim
            from tqdm import tqdm

            print(f"==> [mlx-tune] Loading Multimodal VLM: {model_path}...")
            model_wrapper, processor = FastVisionModel.from_pretrained(
                model_name=model_path,
                load_in_4bit=not qat,
            )

            print(f"==> [mlx-tune] Applying PEFT (audio_layers={tune_audio_encoder}, rank={rank})...")
            model_wrapper = FastVisionModel.get_peft_model(
                model_wrapper,
                finetune_vision_layers=False,
                finetune_language_layers=True,
                finetune_audio_layers=tune_audio_encoder,
                finetune_attention_modules=True,
                finetune_mlp_modules=True,
                r=rank,
                lora_alpha=rank * 2,
                lora_dropout=0.0,
                bias="none",
            )
            FastVisionModel.for_training(model_wrapper)

            # Load dataset from JSONL
            train_jsonl = os.path.join(data_path, "train.jsonl")
            train_dataset = []
            with open(train_jsonl, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        train_dataset.append(json.loads(line))

            print(f"==> Loaded {len(train_dataset)} training examples from {train_jsonl}")

            collator = UnslothVisionDataCollator(model_wrapper, processor)
            optimizer = optim.Adam(learning_rate=learning_rate)
            assistant_id = _detect_assistant_role_token(processor) or 77091
            trainer = _VLMTrainerShim(model_wrapper.model, optimizer, train_on_completions=True, assistant_id=assistant_id)
            loss_and_grad_fn = nn.value_and_grad(trainer.model, trainer.loss_fn)

            progress = tqdm(range(iters), desc="Training Gemma 4 VLM")
            total_loss = 0.0
            step = 0
            step_start = time.time()

            while step < iters:
                for i in range(0, len(train_dataset), batch_size):
                    if step >= iters:
                        break
                    batch_samples = train_dataset[i : i + batch_size]
                    batch = collator(batch_samples)

                    loss, grads = loss_and_grad_fn(trainer.model, batch)
                    trainer.optimizer.update(trainer.model, grads)
                    mx.eval(trainer.model, trainer.optimizer.state)

                    loss_val = float(loss.item())
                    total_loss += loss_val
                    step += 1

                    step_dur = time.time() - step_start
                    step_start = time.time()
                    it_sec = 1.0 / max(step_dur, 1e-4)
                    peak_mem = float(mx.metal.get_peak_memory() / (1024**3)) if mx.metal.is_available() else 0.0

                    progress.update(1)
                    progress.set_postfix({"loss": f"{loss_val:.4f}", "avg_loss": f"{(total_loss/step):.4f}"})

                    entry = {
                        "type": "train",
                        "iter": step,
                        "loss": round(loss_val, 4),
                        "learning_rate": learning_rate,
                        "it_sec": round(it_sec, 2),
                        "peak_mem_gb": round(peak_mem, 3)
                    }
                    with open(log_file, "a") as lf:
                        lf.write(json.dumps(entry) + "\n")

            progress.close()

            # Save adapter artifacts
            adapter_dir = Path("adapters")
            adapter_dir.mkdir(parents=True, exist_ok=True)
            save_adapter(trainer.model, str(adapter_dir / "adapters.safetensors"))

            # Save dual-compatible adapter_config.json
            adapter_cfg = {
                "rank": rank,
                "alpha": float(rank * 2),
                "dropout": 0.0,
                "fine_tune_type": "lora",
                "lora_parameters": {
                    "rank": rank,
                    "scale": 2.0,
                    "dropout": 0.0,
                    "keys": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
                }
            }
            with open(adapter_dir / "adapter_config.json", "w") as acf:
                json.dump(adapter_cfg, acf, indent=2)

            trained_native = True
            print(f"==> Training complete! Adapters saved to {adapter_dir}/")

        else:
            from mlx_tune import FastLanguageModel
            from mlx_lm.tuner.trainer import train as mlx_train, TrainingArgs
            from mlx_lm.tuner.datasets import load_dataset as mlx_load_dataset, CacheDataset
            import mlx.optimizers as optim
            import types

            print(f"==> [mlx-tune] Loading Causal LM: {model_path}...")
            model_wrapper, tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_path,
                max_seq_length=2048,
                load_in_4bit=not qat,
            )
            model_wrapper = FastLanguageModel.get_peft_model(
                model_wrapper,
                r=rank,
                lora_alpha=rank * 2,
                num_layers=lora_layers,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
            )

            # Apply LoRA in memory
            if hasattr(model_wrapper, '_apply_lora') and not model_wrapper._lora_applied:
                model_wrapper._apply_lora(num_layers=lora_layers)

            actual_model = model_wrapper.model if hasattr(model_wrapper, 'model') else model_wrapper

            # Setup dataset
            dataset_args = types.SimpleNamespace(
                data=data_path,
                train=True,
                test=False,
                hf_dataset=None,
                mask_prompt=False,
            )
            train_set, valid_set, _ = mlx_load_dataset(args=dataset_args, tokenizer=tokenizer)
            train_set = CacheDataset(train_set)
            valid_set = CacheDataset(valid_set) if valid_set else None

            # Setup optimizer and arguments
            optimizer = optim.AdamW(learning_rate=learning_rate)
            adapter_dir = Path("adapters")
            adapter_dir.mkdir(parents=True, exist_ok=True)
            adapter_file = str(adapter_dir / "adapters.safetensors")

            training_args = TrainingArgs(
                batch_size=batch_size,
                iters=iters,
                val_batches=5 if valid_set else 0,
                steps_per_report=max(1, iters // 20),
                steps_per_eval=max(iters // 5, 50),
                steps_per_save=iters,
                max_seq_length=2048,
                adapter_file=adapter_file,
                grad_checkpoint=True,
            )

            print("==> Launching native training loop with MLXMonitor streaming...")
            mlx_train(
                model=actual_model,
                optimizer=optimizer,
                train_dataset=train_set,
                val_dataset=valid_set,
                args=training_args,
                training_callback=MLXMonitorCallback(log_file),
            )

            # Save adapter config
            adapter_cfg = {
                "rank": rank,
                "alpha": float(rank * 2),
                "dropout": 0.0,
                "fine_tune_type": "lora",
                "num_layers": lora_layers,
                "lora_parameters": {
                    "rank": rank,
                    "scale": 2.0,
                    "dropout": 0.0,
                    "keys": ["self_attn.q_proj", "self_attn.v_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]
                }
            }
            with open(adapter_dir / "adapter_config.json", "w") as acf:
                json.dump(adapter_cfg, acf, indent=2)

            trained_native = True
            print(f"==> Training complete! Adapters saved to {adapter_dir}/")

    except Exception as e:
        print(f"\n⚠️  Programmatic training encountered an exception: {e}")
        print("==> Engaging robust CLI subprocess fallback pipeline...")

    if not trained_native:
        # Fallback to subprocess training
        if is_vlm:
            cmd = [
                "python", "-m", "mlx_vlm.lora",
                "--model-path", model_path,
                "--dataset", data_path,
                "--iters", str(iters),
                "--batch-size", str(batch_size),
                "--output-path", "adapters",
                "--lora-rank", str(rank),
                "--learning-rate", str(learning_rate)
            ]
        else:
            config_content = f"""# Temporary LoRA Config for MLX
model: "{model_path}"
train: true
data: "{data_path}"
iters: {iters}
batch_size: {batch_size}
num_layers: {lora_layers}
learning_rate: {learning_rate}
grad_checkpoint: true
clear_cache_threshold: 0.1
lora_parameters:
  rank: {rank}
  scale: 20.0
  dropout: 0.0
  keys: ["self_attn.q_proj", "self_attn.v_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]
"""
            with open("temp_lora_config.yaml", "w") as cf:
                cf.write(config_content)
            cmd = ["python", "-m", "mlx_lm.lora", "--config", "temp_lora_config.yaml"]

        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        train_pattern = re.compile(r"Iter\s+(\d+):\s+Train loss\s+([\d.]+),\s+Learning Rate\s+([\d.e+-]+),\s+It/sec\s+([\d.]+),\s+Tokens/sec\s+([\d.]+),\s+Trained Tokens\s+(\d+),\s+Peak mem\s+([\d.]+)\s+GB")
        val_pattern = re.compile(r"Iter\s+(\d+):\s+Val loss\s+([\d.]+),\s+Val took\s+([\d.]+)s")

        for line in process.stdout:
            print(line, end="")
            clean_line = re.sub(r'\x1b\[[0-9;]*m', '', line).strip()
            log_entry = None
            train_match = train_pattern.search(clean_line)
            if train_match:
                log_entry = {
                    "type": "train",
                    "iter": int(train_match.group(1)),
                    "loss": float(train_match.group(2)),
                    "learning_rate": float(train_match.group(3)),
                    "it_sec": float(train_match.group(4)),
                    "tokens_sec": float(train_match.group(5)),
                    "trained_tokens": int(train_match.group(6)),
                    "peak_mem_gb": float(train_match.group(7))
                }
            val_match = val_pattern.search(clean_line)
            if val_match:
                log_entry = {
                    "type": "val",
                    "iter": int(val_match.group(1)),
                    "loss": float(val_match.group(2)),
                    "val_took_s": float(val_match.group(3))
                }
            if log_entry:
                with open(log_file, "a") as f:
                    f.write(json.dumps(log_entry) + "\n")
        process.wait()

    end_time = datetime.datetime.now(datetime.timezone.utc).isoformat()
    with open(log_file, "a") as f:
        f.write(json.dumps({"type": "status", "status": "completed", "timestamp": end_time}) + "\n")


def run_eval(model_path, adapter_path, prompt, multimodal=False):
    """
    Tests the newly trained adapter by loading the base model, 
    injecting the adapter weights in memory, and generating text.
    """
    print(f"==> Evaluating model with mlx-tune...")
    is_vlm = _is_vlm_model(model_path, explicit_multimodal=multimodal)

    try:
        if is_vlm:
            from mlx_tune import FastVisionModel
            model_wrapper, processor = FastVisionModel.from_pretrained(model_path)
            model_wrapper.load_adapter(adapter_path)
            FastVisionModel.for_inference(model_wrapper)
            response = model_wrapper.generate(prompt=prompt, max_tokens=256)
            print("\n==> Model Response:")
            print(response)
            return
        else:
            from mlx_tune import FastLanguageModel
            model_wrapper, tokenizer = FastLanguageModel.from_pretrained(model_path)
            model_wrapper.load_adapter(adapter_path)
            FastLanguageModel.for_inference(model_wrapper)

            # Apply native chat template for instruction-tuned models
            eval_prompt = prompt
            if hasattr(tokenizer, "apply_chat_template"):
                try:
                    messages = [{"role": "user", "content": prompt}]
                    eval_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                except Exception:
                    pass

            response = model_wrapper.generate(prompt=eval_prompt, max_tokens=256)
            print("\n==> Model Response:")
            print(response)
            return
    except Exception as e:
        print(f"Note: Programmatic eval fallback ({e}); using CLI...")

    if is_vlm:
        cmd = ["python", "-m", "mlx_vlm.generate", "--model", model_path, "--adapter-path", adapter_path, "--prompt", prompt, "--max-tokens", "200"]
    else:
        cmd = ["python", "-m", "mlx_lm.generate", "--model", model_path, "--adapter-path", adapter_path, "--prompt", prompt, "--max-tokens", "200"]
    subprocess.run(cmd)
    subprocess.run(cmd)

def _fuse_multimodal(model_path, adapter_path, save_path):
    """Fuse a Gemma 4 (VLM / audio) LoRA into the base and save full-precision weights.

    Loads the base with mlx-tune's FastVisionModel, applies the adapter with the
    same LoRA layers training used (load_adapter), then replaces every LoRALinear
    (language model and, if tuned, audio tower) with its dequantized fused Linear
    via LoRALinear.fuse(dequantize=True). Weights are saved under the base
    checkpoint's key layout (model.language_model.*, model.audio_tower.*, ...), and
    the base's config/processor/tokenizer files are copied, so the output loads
    like the original checkpoint.
    """
    import glob
    import mlx.core as mx
    from mlx.utils import tree_flatten, tree_unflatten
    from mlx_tune import FastVisionModel

    wrapper, processor = FastVisionModel.from_pretrained(model_path)
    if adapter_path:
        wrapper.load_adapter(adapter_path)
    model = wrapper.model

    if adapter_path:
        # mlx-tune's load_adapter only rebuilds language-model LoRA. Adapters trained
        # with --tune-audio-encoder also carry audio-tower LoRA (on ClippableLinear.linear)
        # plus trained full tensors (audio clip bounds, per_dim_scale, embed_audio);
        # rebuild/apply those here so they are not silently dropped.
        from mlx_lm.tuner.lora import LoRALinear
        saved = mx.load(os.path.join(adapter_path, "adapters.safetensors"))
        with open(os.path.join(adapter_path, "adapter_config.json")) as f:
            cfg = json.load(f)
        lp = cfg.get("lora_parameters", {})
        scale = float(lp.get("scale", cfg.get("alpha", 16.0) / max(cfg.get("rank", 8), 1)))
        audio_lora = sorted({k[: -len(".lora_a")] for k in saved
                             if k.startswith("audio_tower.") and k.endswith(".lora_a")})
        for path in audio_lora:
            parts = path.split(".")
            parent = model
            for part in parts[:-1]:
                parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
            base = getattr(parent, parts[-1])
            if type(base).__name__ != "LoRALinear":
                lora_a = saved[path + ".lora_a"]
                setattr(parent, parts[-1], LoRALinear.from_base(base, r=lora_a.shape[1], scale=scale))
        extras = [(k, v) for k, v in saved.items()
                  if k.startswith(("audio_tower.", "embed_audio.")) and not k.endswith((".lora_a", ".lora_b"))]
        audio_lora_weights = [(k, v) for k, v in saved.items()
                              if k.startswith("audio_tower.") and k.endswith((".lora_a", ".lora_b"))]
        if audio_lora or extras:
            model.load_weights(audio_lora_weights + extras, strict=False)
            print(f"==> Audio tower: {len(audio_lora)} LoRA layers + {len(extras)} trained tensors applied")

    fused = [(name, module.fuse(dequantize=True))
             for name, module in model.named_modules()
             if type(module).__name__ == "LoRALinear" and hasattr(module, "fuse")]
    if adapter_path and not fused:
        raise SystemExit(f"Error: no LoRA layers were applied from {adapter_path}; nothing to fuse.")
    if fused:
        model.update_modules(tree_unflatten(fused))
        print(f"==> Fused {len(fused)} LoRA layers (dequantized)")

    weights = {}
    for key, value in tree_flatten(model.parameters()):
        hf_key = "model." + key
        hf_key = hf_key.replace("model.language_model.model.", "model.language_model.")
        weights[hf_key] = value

    os.makedirs(save_path, exist_ok=True)
    mx.save_safetensors(os.path.join(save_path, "model.safetensors"), weights)
    for pattern in ("*.json", "*.jinja", "tokenizer.model"):
        for src in glob.glob(os.path.join(model_path, pattern)):
            if os.path.basename(src) != "model.safetensors.index.json":
                shutil.copy2(src, save_path)
    print(f"==> Multimodal fusion complete: {len(weights)} tensors -> {save_path}")


def run_fuse(model_path, adapter_path, save_path, multimodal=False):
    """
    Permanently bakes the trained LoRA adapters into the base model and writes
    full-precision (dequantized) HF-format weights, readable by llama.cpp and
    mlx_lm convert.
    """
    print(f"==> Fusing LoRA adapters into base model (dequantizing for GGUF compatibility)...")
    if multimodal:
        _fuse_multimodal(model_path, adapter_path, save_path)
        return

    # Text models: mlx-tune load + load_adapter + save_pretrained_merged(merged_16bit)
    # (fuses LoRA and dequantizes to full precision, ready for llama.cpp / mlx_lm convert).
    from mlx_tune import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(model_path)
    if adapter_path:
        model.load_adapter(adapter_path)
    model.save_pretrained_merged(save_path, tokenizer, save_method="merged_16bit")

def _default_llama_cpp_path():
    """$LLAMA_CPP_PATH, else the llama.cpp checkout at the gemmma repo root (if present)."""
    env = os.environ.get("LLAMA_CPP_PATH")
    if env:
        return env
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    candidate = os.path.join(repo_root, "llama.cpp")
    return candidate if os.path.isdir(candidate) else None


def _converter_python(llama_cpp_path):
    """Python for convert_hf_to_gguf.py: $LLAMA_CPP_PYTHON, else <llama.cpp>/.venv, else current.

    llama.cpp pins its converter deps (e.g. transformers==5.5.1); newer transformers
    drop Gemma 4's default global_head_dim from the config and the converter fails.
    Create it once with:
      uv venv --python 3.12 llama.cpp/.venv
      uv pip install --python llama.cpp/.venv/bin/python \\
          -r llama.cpp/requirements/requirements-convert_hf_to_gguf.txt --index-strategy unsafe-best-match
    """
    env = os.environ.get("LLAMA_CPP_PYTHON")
    if env:
        return env
    if llama_cpp_path:
        candidate = os.path.join(llama_cpp_path, ".venv", "bin", "python")
        if os.path.exists(candidate):
            return candidate
    return None


def run_gguf(model_path, output_path, outtype="q8_0", qat=False, adapter_path=None, llama_cpp_path=None):
    """
    Export a model to a single GGUF file via mlx-tune's llama.cpp pipeline
    (fuse + dequantize -> convert_hf_to_gguf.py -> llama-quantize).

    model_path is either an already-fused model directory, or a base model
    combined with adapter_path. Tokenizer files are taken only from model_path
    (never copied in from elsewhere; see FL-005).
    """
    from mlx_tune import export_to_gguf, LlamaCppNotFoundError

    if qat and outtype != "q4_0":
        print(f"==> [QAT Optimization] Overriding outtype '{outtype}' to 'q4_0' for QAT alignment.")
        print("    Gemma 4 QAT weights are pre-conditioned for q4_0; other types re-round them.")

    llama_cpp_path = llama_cpp_path or _default_llama_cpp_path()
    converter_python = _converter_python(llama_cpp_path)
    if converter_python is None:
        print("==> Warning: no llama.cpp/.venv found; running the converter with this interpreter.")
        print("    If conversion fails (e.g. KeyError: 'global_head_dim'), create the converter venv:")
        print("    see _converter_python() in cli.py or docs/FRICTION_LOG.md FL-007.")
    try:
        export_to_gguf(
            model_path,
            output_path=output_path,
            quantization=outtype,
            adapter_path=adapter_path,
            qat=qat,
            llama_cpp_path=llama_cpp_path,
            llama_cpp_python=converter_python,
        )
    except LlamaCppNotFoundError as e:
        print(f"Error: {e}")
        raise SystemExit(1)

def run_benchmark(reference_model_path, gguf_model_path, prompt, multimodal=False):
    """
    Benchmarks and compares the outputs of the high-precision fused model (reference)
    and the quantized GGUF model to measure 'Quantization Drift'.
    Also provides instructions on how to use llama-perplexity for formal validation.
    """
    import sys
    import re
    
    print("\n" + "="*80)
    print("==> STARTING QUANTIZATION DRIFT BENCHMARK")
    print("="*80)
    print(f"Reference FP16 Model: {reference_model_path}")
    print(f"Quantized GGUF Model:   {gguf_model_path}")
    print(f"Evaluation Prompt:      '{prompt}'\n")

    # 1. Run inference on high-precision reference model using MLX
    print("==> Step 1: Generating high-precision FP16 reference completion using MLX...")
    ref_response = ""
    
    try:
        if multimodal:
            from mlx_vlm import load as load_vlm, generate as generate_vlm
            model, processor = load_vlm(reference_model_path)
            # Standard simple prompt formatting
            formatted_prompt = f"User: {prompt}\nAssistant:"
            res = generate_vlm(model, processor, prompt=formatted_prompt, max_tokens=150, verbose=False)
            ref_response = res.text if hasattr(res, "text") else res
        else:
            from mlx_lm import load as load_lm, generate as generate_lm
            model, tokenizer = load_lm(reference_model_path)
            formatted_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
            res = generate_lm(model, tokenizer, prompt=formatted_prompt, max_tokens=150, verbose=False)
            ref_response = res.text if hasattr(res, "text") else res
        
        print("\n--- Reference FP16 Output ---")
        print(ref_response.strip())
        print("-" * 30 + "\n")
    except Exception as e:
        print(f"Error loading/running reference model: {e}")
        print("Please ensure the reference model has been fused and saved to the specified directory.")
        return

    # 2. Run inference on quantized GGUF model using llama-cli
    print("==> Step 2: Generating low-precision quantized completion using llama-cli...")
    gguf_response = ""
    
    # Locate llama-cli
    llama_cli_path = shutil.which("llama-cli") or shutil.which("llama-main") or "/opt/homebrew/bin/llama-cli"
    if not os.path.exists(llama_cli_path) and not shutil.which("llama-cli"):
        print("Warning: Could not find 'llama-cli' in your PATH.")
        print("Please ensure you have run 'brew install llama.cpp' or compiled llama.cpp locally.")
        print("Skipping GGUF generation. Displaying educational guide instead...\n")
    else:
        cmd = [
            llama_cli_path,
            "-m", gguf_model_path,
            "-p", f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n",
            "-n", "150",
            "--quiet"
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            gguf_response = result.stdout.strip()
            # Clean up the output to exclude the prompt if llama-cli prints it
            prompt_marker = "<|im_start|>assistant\n"
            if prompt_marker in gguf_response:
                gguf_response = gguf_response.split(prompt_marker)[-1]
            
            print("\n--- Quantized GGUF Output ---")
            print(gguf_response.strip())
            print("-" * 30 + "\n")
        except Exception as e:
            print(f"Error running llama-cli: {e}")
            print("This could be due to model configuration or GGUF path issues.\n")

    # 3. Calculate Quantization Drift using Jaccard Similarity on word level
    if ref_response and gguf_response:
        print("==> Step 3: Quantization Drift Analysis")
        
        def tokenize(text):
            # Lowercase and clean punctuation for robust word comparison
            words = re.findall(r'\w+', text.lower())
            return set(words)
            
        ref_tokens = tokenize(ref_response)
        gguf_tokens = tokenize(gguf_response)
        
        intersection = ref_tokens.intersection(gguf_tokens)
        union = ref_tokens.union(gguf_tokens)
        
        jaccard_score = len(intersection) / len(union) if union else 0.0
        drift_score = 1.0 - jaccard_score
        
        print(f"    * Vocabulary Jaccard Similarity (Word-level overlap): {jaccard_score:.2%}")
        print(f"    * Estimated Quantization Drift (Semantic Divergence):  {drift_score:.2%}")
        print("\n    Note: Lower Quantization Drift means the low-precision model behaves almost")
        print("          identically to the high-precision reference model. QAT pre-conditioned")
        print("          models typically achieve much higher similarity scores compared to PTQ models.\n")

    # 4. Educational guide on llama-perplexity
    print("="*80)
    print("==> HOW TO RUN MATHEMATICAL VALIDATION (PEDAGOGY GUIDE)")
    print("="*80)
    print("To quantitatively evaluate how much accuracy was saved by using QAT vs. Standard models,")
    print("you should compute mathematical PERPLEXITY. Lower perplexity is better.\n")
    print("Run these commands in your terminal:\n")
    print("  # 1. Measure perplexity of your standard PTQ model (e.g. standard Gemma 4 quantized to q4_0)")
    print("  llama-perplexity -m standard-q4_0.gguf -f wikitext-2-raw/wiki.test.raw\n")
    print("  # 2. Measure perplexity of your QAT model (Gemma 4 QAT quantized to q4_0)")
    print("  llama-perplexity -m qat-q4_0.gguf -f wikitext-2-raw/wiki.test.raw\n")
    print("  *Observation:* You will find that the QAT model has a perplexity score much closer")
    print("  to the unquantized FP16 baseline than the standard PTQ model does!")
    print("="*80 + "\n")

def run_clean():
    """
    Cleans up temporary artifacts from previous training runs.
    """
    print("==> Cleaning up training artifacts...")
    dirs_to_remove = ["adapters", "fused_model_dequantized"]
    files_to_remove = ["training_log.jsonl"]
    
    for d in dirs_to_remove:
        if os.path.exists(d):
            shutil.rmtree(d)
            print(f"    Removed directory: {d}/")
            
    for f in files_to_remove:
        if os.path.exists(f):
            os.remove(f)
            print(f"    Removed file: {f}")
            
    print("==> Workspace is clean and ready for a new training run.")

def render_journal_md(db, md_path):
    """
    Generates a beautiful Markdown journal file from the experiments dictionary.
    """
    content = """# 📓 Gemma 4 QAT Fine-Tuning Experiment Journal

This journal serves as an experiment tracker and MLOps log for training, fusing, quantizing, and evaluating Gemma 4 QAT (Quantization-Aware Training) models on Apple Silicon.

---
"""
    
    # Sort experiments by ID or date, descending (newest first)
    sorted_experiments = sorted(db.items(), key=lambda x: x[1].get("date", ""), reverse=True)
    
    if not sorted_experiments:
        content += "\n*No experiments logged yet. Run `mlxtune journal init --id QAT-001` to start tracking!*\n"
    else:
        for eid, exp in sorted_experiments:
            status = exp["status"]
            status_color = "🟡 PLANNED" if status == "PLANNED" else "🟡 IN PROGRESS" if status == "IN_PROGRESS" else "🟢 COMPLETED" if status == "COMPLETED" else "🔴 FAILED"
            
            hp = exp["hyperparameters"]
            res = exp["results"]
            
            content += f"""
## 🔬 Experiment ID: {eid}
*   **Date:** {exp.get('date', 'unknown')}
*   **Base Model:** `{exp.get('model', 'unknown')}`
*   **Hardware:** Apple Silicon Mac (Metal GPU)
*   **Status:** {status_color}

---

### 🎛️ Hyperparameters

| Parameter | Value | Rationale / Detail |
| :--- | :--- | :--- |
| **Iterations (`--iters`)** | `{hp.get('iters') or 'N/A'}` | Training steps. |
| **Batch Size** | `{hp.get('batch_size') or 'N/A'}` | Batch training count. |
| **LoRA Rank (`--rank`)** | `{hp.get('rank') or 'N/A'}` | LoRA adapter width. |
| **Target Layers** | `{hp.get('layers') or 'N/A'}` | Frozen boundary layer targeting. |
| **Dataset** | `{hp.get('dataset') or 'N/A'}` | Data source. |

---

### 📊 Evaluation & Results

#### 1. Training Telemetry
*   **Validation Loss at start:** `{res.get('val_loss_start') if res.get('val_loss_start') is not None else '*[Awaiting execution]*'}`
*   **Validation Loss at end:** `{res.get('val_loss_end') if res.get('val_loss_end') is not None else '*[Awaiting execution]*'}`
*   **Peak GPU Memory:** `{f"{res.get('peak_mem_gb')} GB" if res.get('peak_mem_gb') is not None else '*[Awaiting execution]*'}`

#### 2. Quantization Drift Analysis
"""
            if res.get("ref_response"):
                content += f"""*   **Reference FP16 Response:**
    ```text
    {res.get('ref_response').strip()}
    ```
"""
            else:
                content += "*   **Reference FP16 Response:** *[Awaiting execution]*\n"
                
            if res.get("gguf_response"):
                content += f"""*   **Quantized GGUF Response:**
    ```text
    {res.get('gguf_response').strip()}
    ```
"""
            else:
                content += "*   **Quantized GGUF Response:** *[Awaiting execution]*\n"
                
            jaccard = res.get("jaccard_score")
            drift = res.get("drift_score")
            
            content += f"""*   **Jaccard Similarity Score:** `{f"{jaccard:.2%}" if jaccard is not None else '*[Awaiting execution]*'}`
*   **Calculated Quantization Drift:** `{f"{drift:.2%}" if drift is not None else '*[Awaiting execution]*'}`
"""
            
            if exp.get("notes"):
                content += f"""
---

### 💡 Notes & Lessons Learned
{exp.get('notes')}
"""
            content += "\n---\n"
            
    with open(md_path, "w") as f:
        f.write(content)
    print(f"==> Rendered beautiful journal markdown to '{md_path}'")


def run_journal(args):
    """
    Manages the Experiment Journal database (JSON) and renders it to a beautiful Markdown file.
    """
    import datetime
    db_path = "docs/experiments.json"
    md_path = "docs/EXPERIMENT_JOURNAL.md"
    
    # Load or initialize JSON database
    if os.path.exists(db_path):
        try:
            with open(db_path, "r") as f:
                db = json.load(f)
        except Exception:
            db = {}
    else:
        db = {}
        
    if args.action == "init":
        if not args.id:
            print("Error: --id is required for init")
            return
        
        # Check if experiment exists
        if args.id in db:
            print(f"Warning: Experiment {args.id} already exists. Initializing will overwrite its hyperparameters.")
            
        db[args.id] = {
            "date": datetime.datetime.now().strftime("%Y-%m-%d"),
            "model": args.model or "unknown",
            "status": args.status or "PLANNED",
            "hyperparameters": {
                "iters": args.iters or 0,
                "batch_size": args.batch_size or 0,
                "rank": args.rank or 8,
                "layers": args.layers or 16,
                "dataset": args.dataset or "unknown"
            },
            "results": {
                "val_loss_start": args.val_start,
                "val_loss_end": args.val_end,
                "peak_mem_gb": args.peak_mem,
                "jaccard_score": args.jaccard,
                "drift_score": args.drift,
                "ref_response": args.ref_response,
                "gguf_response": args.gguf_response
            },
            "notes": args.notes or ""
        }
        print(f"==> Initialized experiment '{args.id}' in journal database.")
        
    elif args.action == "update":
        if not args.id:
            print("Error: --id is required for update")
            return
            
        if args.id not in db:
            print(f"Error: Experiment '{args.id}' not found. Run 'init' first.")
            return
            
        exp = db[args.id]
        if args.status:
            exp["status"] = args.status
        if args.model:
            exp["model"] = args.model
            
        # Update hyperparameters if provided
        if args.iters is not None: exp["hyperparameters"]["iters"] = args.iters
        if args.batch_size is not None: exp["hyperparameters"]["batch_size"] = args.batch_size
        if args.rank is not None: exp["hyperparameters"]["rank"] = args.rank
        if args.layers is not None: exp["hyperparameters"]["layers"] = args.layers
        if args.dataset is not None: exp["hyperparameters"]["dataset"] = args.dataset
        
        # Update results if provided
        if args.val_start is not None: exp["results"]["val_loss_start"] = args.val_start
        if args.val_end is not None: exp["results"]["val_loss_end"] = args.val_end
        if args.peak_mem is not None: exp["results"]["peak_mem_gb"] = args.peak_mem
        if args.jaccard is not None: exp["results"]["jaccard_score"] = args.jaccard
        if args.drift is not None: exp["results"]["drift_score"] = args.drift
        if args.ref_response is not None: exp["results"]["ref_response"] = args.ref_response
        if args.gguf_response is not None: exp["results"]["gguf_response"] = args.gguf_response
        if args.notes is not None:
            if exp["notes"]:
                exp["notes"] += "\n" + args.notes
            else:
                exp["notes"] = args.notes
                
        print(f"==> Updated experiment '{args.id}' in journal database.")
        
    elif args.action == "list":
        print("\n" + "="*80)
        print("==> EXPERIMENT JOURNAL DIRECTORY")
        print("="*80)
        if not db:
            print("No experiments logged yet.")
        for eid, exp in db.items():
            status_emoji = "🟢" if exp["status"] == "COMPLETED" else "🟡" if exp["status"] in ["PLANNED", "IN_PROGRESS"] else "🔴"
            print(f"{status_emoji} {eid:<18} | Date: {exp['date']} | Model: {exp['model'][:40]:<40} | Status: {exp['status']}")
        print("="*80 + "\n")
        return
        
    elif args.action == "rebuild":
        print("==> Rebuilding experiment journal markdown from database...")
        
    # Save the updated database
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    with open(db_path, "w") as f:
        json.dump(db, f, indent=2)
        
    # Render to Markdown
    render_journal_md(db, md_path)


def main():
    parser = argparse.ArgumentParser(description="MLX LoRA Fine-Tuning Easy Button")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # 1. Download Command
    download_parser = subparsers.add_parser("download", help="Download base MLX model")
    download_parser.add_argument("--model", default="mlx-community/gemma-2-2b-it-4bit", help="Hugging Face model ID")
    download_parser.add_argument("--dest", default="./model", help="Destination folder")

    # 2 & 3. Prep Data Command
    prep_parser = subparsers.add_parser("prep", help="Download and prepare dataset")
    prep_parser.add_argument("--dataset", default="yahma/alpaca-cleaned", help="Hugging Face dataset ID")
    prep_parser.add_argument("--dataset-config", default=None, help="Hugging Face dataset configuration name (e.g. en-US)")
    prep_parser.add_argument("--dest", default="./data", help="Destination folder for JSONL files")
    prep_parser.add_argument("--samples", type=int, default=1000, help="Max samples to use for fast training")

    # 4. Train Command
    train_parser = subparsers.add_parser("train", help="Run LoRA training")
    train_parser.add_argument("--model", default="./model", help="Path to base model")
    train_parser.add_argument("--data", default="./data", help="Path to data folder")
    train_parser.add_argument("--iters", type=int, default=200, help="Number of training iterations (low for testing)")
    train_parser.add_argument("--batch-size", type=int, default=2, help="Batch size")
    train_parser.add_argument("--log-file", default="training_log.jsonl", help="File to output JSONL metrics")
    train_parser.add_argument("--multimodal", action="store_true", help="Use mlx_vlm for multimodal training")
    train_parser.add_argument("--tune-audio-encoder", action="store_true", help="Target Audio Encoder for Multimodal LoRA")
    train_parser.add_argument("--qat", action="store_true", help="Enable Quantization-Aware Training (QAT) optimizations")
    train_parser.add_argument("--rank", type=int, default=8, help="LoRA rank (keep low for QAT models to prevent adapter quantization noise)")
    train_parser.add_argument("--lora-layers", type=int, default=16, help="Number of LoRA layers to target")
    train_parser.add_argument("--lr", "--learning-rate", type=float, default=1e-5, help="Learning rate (default: 1e-5)")

    # 5. Eval Command
    eval_parser = subparsers.add_parser("eval", help="Test the fine-tuned model")
    eval_parser.add_argument("--model", default="./model", help="Path to base model")
    eval_parser.add_argument("--adapter", default="./adapters", help="Path to trained adapter directory")
    eval_parser.add_argument("--prompt", required=True, help="Text prompt to test")
    eval_parser.add_argument("--multimodal", action="store_true", help="Use multimodal evaluation")

    # 6. Fuse Command
    fuse_parser = subparsers.add_parser("fuse", help="Fuse adapters into base model")
    fuse_parser.add_argument("--model", default="./model", help="Path to base model")
    fuse_parser.add_argument("--adapter", default="./adapters", help="Path to trained adapter directory")
    fuse_parser.add_argument("--dest", default="./fused_model_dequantized", help="Destination folder for fused model")
    fuse_parser.add_argument("--multimodal", action="store_true", help="Use mlx_vlm for multimodal fusing")

    # 7. GGUF Export Command
    gguf_parser = subparsers.add_parser("gguf", help="Export a (fused, or base + adapter) model to GGUF via llama.cpp")
    gguf_parser.add_argument("--model", default="./fused_model_dequantized", help="Fused model folder, or base model when --adapter is given")
    gguf_parser.add_argument("--adapter", default=None, help="Optional LoRA adapter folder to fuse into --model before export")
    gguf_parser.add_argument("--dest", default="my-custom-model.gguf", help="Destination GGUF file path")
    gguf_parser.add_argument("--outtype", default="q8_0", help="Quantization type for GGUF (e.g., q8_0, f16, q4_k_m, q4_0)")
    gguf_parser.add_argument("--qat", action="store_true", help="Enforce q4_0 quantization optimized for QAT models")
    gguf_parser.add_argument("--llama-cpp", default=None, help="llama.cpp checkout (default: $LLAMA_CPP_PATH, else ./llama.cpp in this repo)")

    # 8. Clean Command
    clean_parser = subparsers.add_parser("clean", help="Clean up training artifacts to start fresh")

    # 9. Benchmark Command
    benchmark_parser = subparsers.add_parser("benchmark", help="Benchmark and compare quantization drift between standard and QAT models")
    benchmark_parser.add_argument("--reference-model", default="./fused_model_dequantized", help="Path to high-precision reference model (fused FP16 folder)")
    benchmark_parser.add_argument("--gguf-model", default="my-custom-model.gguf", help="Path to quantized GGUF model")
    benchmark_parser.add_argument("--prompt", default="Describe the core benefits of using Quantization-Aware Training (QAT) for edge deployments.", help="Text prompt to test")
    benchmark_parser.add_argument("--multimodal", action="store_true", help="Use multimodal (mlx_vlm) for reference evaluation")

    # 10. Journal Command
    journal_parser = subparsers.add_parser("journal", help="Manage the Experiment Journal")
    journal_parser.add_argument("action", choices=["init", "update", "list", "rebuild"], help="Journal action")
    journal_parser.add_argument("--id", help="Experiment ID (e.g. QAT-DRYRUN-001)")
    journal_parser.add_argument("--model", help="Base model identifier")
    journal_parser.add_argument("--iters", type=int, help="Number of training iterations")
    journal_parser.add_argument("--batch-size", type=int, help="Batch size")
    journal_parser.add_argument("--rank", type=int, help="LoRA rank")
    journal_parser.add_argument("--layers", type=int, help="Number of layers targeted")
    journal_parser.add_argument("--dataset", help="Dataset identifier")
    journal_parser.add_argument("--status", choices=["PLANNED", "IN_PROGRESS", "COMPLETED", "FAILED"], help="Experiment status")
    journal_parser.add_argument("--val-start", type=float, help="Validation loss at start")
    journal_parser.add_argument("--val-end", type=float, help="Validation loss at end")
    journal_parser.add_argument("--peak-mem", type=float, help="Peak GPU memory in GB")
    journal_parser.add_argument("--jaccard", type=float, help="Word-level Jaccard similarity score")
    journal_parser.add_argument("--drift", type=float, help="Semantic drift score")
    journal_parser.add_argument("--ref-response", help="Inference output of the high-precision reference model")
    journal_parser.add_argument("--gguf-response", help="Inference output of the quantized GGUF model")
    journal_parser.add_argument("--notes", help="Experimental observations and notes")

    args = parser.parse_args()

    if args.command == "download":
        download_model(args.model, args.dest)
    elif args.command == "prep":
        prep_dataset(args.dataset, args.dataset_config, args.dest, args.samples)
    elif args.command == "train":
        run_train(args.model, args.data, args.iters, args.batch_size, args.log_file, args.multimodal, args.tune_audio_encoder, args.qat, args.rank, args.lora_layers, args.lr)
    elif args.command == "eval":
        run_eval(args.model, args.adapter, args.prompt, args.multimodal)
    elif args.command == "fuse":
        run_fuse(args.model, args.adapter, args.dest, args.multimodal)
    elif args.command == "gguf":
        run_gguf(args.model, args.dest, args.outtype, args.qat, adapter_path=args.adapter, llama_cpp_path=args.llama_cpp)
    elif args.command == "clean":
        run_clean()
    elif args.command == "benchmark":
        run_benchmark(args.reference_model, args.gguf_model, args.prompt, args.multimodal)
    elif args.command == "journal":
        run_journal(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()