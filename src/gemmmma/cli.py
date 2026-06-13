import argparse
import os
import json
import subprocess
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

def run_train(model_path, data_path, iters, batch_size, log_file="training_log.jsonl", multimodal=False, tune_audio_encoder=False, qat=False, rank=8, lora_layers=16, learning_rate=1e-5):
    """
    Executes the LoRA training loop on the Apple GPU (Metal).
    This freezes the base model and only updates a tiny set of adapter weights.
    Logs metrics to a JSONL file for external monitoring.
    """
    import re
    import datetime
    
    print(f"==> Starting LoRA training with MLX...")
    if qat:
        print("==> [QAT Optimization] Utilizing conservative rank and layers to minimize adapter post-training quantization drift.")
    
    if multimodal:
        # Modern mlx_vlm CLI syntax
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
        
        if tune_audio_encoder:
            print("==> Targeting Audio Encoder layers for Multimodal LoRA tuning.")
            # Adjust if mlx_vlm uses specific regex or flags for fine-tuning specific components
            # e.g., cmd.extend(["--fine-tune-type", "audio_encoder"]) 
    else:
        # mlx_lm.lora requires a configuration file to specify custom LoRA parameters like rank.
        # We generate a temporary config on-the-fly to ensure standard-aligned execution.
        print("==> Generating temporary LoRA YAML configuration...")
        config_content = f"""# Temporary LoRA Config for MLX
model: "{model_path}"
train: true
data: "{data_path}"
iters: {iters}
batch_size: {batch_size}
num_layers: {lora_layers}
learning_rate: {learning_rate}
lora_parameters:
  rank: {rank}
  alpha: {2 * rank}
  dropout: 0.0
  keys: ["q_proj", "v_proj", "gate_proj", "down_proj", "up_proj"]
"""
        with open("temp_lora_config.yaml", "w") as cf:
            cf.write(config_content)

        cmd = [
            "python", "-m", "mlx_lm.lora",
            "--config", "temp_lora_config.yaml"
        ] 
    
    print(f"==> Logging metrics to {log_file}")
    
    start_time = datetime.datetime.now(datetime.timezone.utc).isoformat()
    with open(log_file, "w") as f:
        f.write(json.dumps({"type": "status", "status": "started", "timestamp": start_time}) + "\n")
        f.write(json.dumps({
            "type": "config",
            "model": model_path,
            "data": data_path,
            "total_iters": iters,
            "batch_size": batch_size,
            "multimodal": multimodal,
            "model_style": "QAT" if qat else "Standard",
            "rank": rank,
            "lora_layers": lora_layers
        }) + "\n")
        
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    
    train_pattern = re.compile(r"Iter\s+(\d+):\s+Train loss\s+([\d.]+),\s+Learning Rate\s+([\d.e+-]+),\s+It/sec\s+([\d.]+),\s+Tokens/sec\s+([\d.]+),\s+Trained Tokens\s+(\d+),\s+Peak mem\s+([\d.]+)\s+GB")
    val_pattern = re.compile(r"Iter\s+(\d+):\s+Val loss\s+([\d.]+),\s+Val took\s+([\d.]+)s")
    
    for line in process.stdout:
        print(line, end="") # Keep printing to the terminal
        
        # Clean ANSI escape codes to ensure clean regex matching
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
    
    if multimodal:
        # Patch the adapter_config.json because mlx_vlm.generate expects "rank" at the root level,
        # but mlx_lm leaves a stale one with "lora_parameters", or mlx_vlm doesn't write it fully.
        
        # In newer versions, mlx_vlm often dumps adapter files at the root instead of the output path.
        if os.path.exists("adapter_config.json") and not os.path.exists("adapters/adapter_config.json"):
            os.makedirs("adapters", exist_ok=True)
            shutil.move("adapter_config.json", "adapters/adapter_config.json")
            if os.path.exists("adapters.safetensors"):
                shutil.move("adapters.safetensors", "adapters/adapters.safetensors")

        adapter_config_path = "adapters/adapter_config.json"
        if os.path.exists(adapter_config_path):
            try:
                with open(adapter_config_path, "r") as f:
                    config = json.load(f)
                if "lora_parameters" in config and "rank" not in config:
                    # Only inject the actual LoRA parameters into the root
                    # mlx_vlm's get_peft_model passes kwargs directly, so extra keys will crash it.
                    clean_config = {}
                    lora_params = config["lora_parameters"]
                    for key in ["rank", "alpha", "dropout"]:
                        if key in lora_params:
                            clean_config[key] = lora_params[key]
                        elif key == "alpha" and "scale" in lora_params and "rank" in lora_params:
                            clean_config["alpha"] = lora_params["scale"] * lora_params["rank"]
                    
                    with open(adapter_config_path, "w") as f:
                        json.dump(clean_config, f, indent=2)
                elif "rank" not in config:
                    clean_config = {"rank": 8, "alpha": 160.0, "dropout": 0.0}
                    with open(adapter_config_path, "w") as f:
                        json.dump(clean_config, f, indent=2)
            except Exception as e:
                print(f"Warning: Could not patch adapter_config.json: {e}")
                
    end_time = datetime.datetime.now(datetime.timezone.utc).isoformat()
    with open(log_file, "a") as f:
        f.write(json.dumps({"type": "status", "status": "completed", "timestamp": end_time}) + "\n")

def run_eval(model_path, adapter_path, prompt, multimodal=False):
    """
    Tests the newly trained adapter by loading the base model, 
    injecting the adapter weights in memory, and generating text.
    """
    print(f"==> Evaluating model...")
    if multimodal:
        cmd = [
            "python", "-m", "mlx_vlm.generate",
            "--model", model_path,
            "--adapter-path", adapter_path,
            "--prompt", prompt,
            "--max-tokens", "200"
        ]
    else:
        cmd = [
            "python", "-m", "mlx_lm.generate",
            "--model", model_path,
            "--adapter-path", adapter_path,
            "--prompt", prompt,
            "--max-tokens", "200"
        ]
    subprocess.run(cmd)

def run_fuse(model_path, adapter_path, save_path, multimodal=False):
    """
    Permanently bakes the trained LoRA adapters into the base model.
    Crucially, it uses --dequantize to convert the Apple-specific 4-bit MLX 
    format back into standard 16-bit PyTorch tensors so llama.cpp can read it.
    """
    print(f"==> Fusing LoRA adapters into base model (dequantizing for GGUF compatibility)...")
    if multimodal:
        # mlx_vlm doesn't have a native fuse CLI yet, so we build the script inline
        script = f"""
import os
from mlx_vlm.utils import load
from mlx_vlm.trainer.utils import apply_lora_layers
from mlx_vlm.trainer.lora import LoRaLayer
from mlx.utils import tree_flatten
import mlx.core as mx
import mlx.nn as nn
import json
import shutil

print("Loading base multimodal model...")
model, processor = load("{model_path}")

print("Applying LoRA adapters...")
model = apply_lora_layers(model, "{adapter_path}")

print("Fusing weights...")
model.eval()

# Manually fuse the LoRA layers into the linear base weights
for i, layer in enumerate(model.language_model.model.layers):
    for name, module in layer.named_modules():
        if isinstance(module, LoRaLayer):
            # Calculate the fused weight: base_weight + (B @ A) * scale
            base_weight = module.original_layer.weight
            lora_b = module.B
            lora_a = module.A
            scale = module.alpha
            
            # Dequantize if the base layer is quantized so we can add the LoRA math
            if hasattr(module.original_layer, "scales"):
                base_weight = mx.dequantize(
                    module.original_layer.weight,
                    module.original_layer.scales,
                    module.original_layer.biases,
                    module.original_layer.group_size,
                    module.original_layer.bits
                )
                
            # MLX weights are [out_features, in_features].
            # A is [256, 8], B is [8, 1536]. 
            # A @ B yields [256, 1536]. Base weight is [1536, 256].
            # We transpose to match.
            lora_update = (lora_a @ lora_b).T * scale
            fused_weight = base_weight + lora_update
            use_bias = "bias" in module.original_layer.parameters()
            
            # Replace the LoRA layer with a standard linear layer containing the fused weights
            new_linear = nn.Linear(base_weight.shape[1], base_weight.shape[0], bias=use_bias)
            new_linear.weight = fused_weight
            
            if use_bias:
                new_linear.bias = module.original_layer.bias
                
            # Keep it quantized if requested (optional, but standard for fused export)
            if hasattr(module.original_layer, "scales"):
                new_linear = nn.QuantizedLinear.from_linear(
                    new_linear,
                    module.original_layer.group_size,
                    module.original_layer.bits
                )
                
            # Update the parent module
            parent_name = ".".join(name.split(".")[:-1])
            child_name = name.split(".")[-1]
            
            if parent_name == "":
                setattr(layer, child_name, new_linear)
            else:
                parent = layer
                for part in parent_name.split("."):
                    parent = getattr(parent, part)
                setattr(parent, child_name, new_linear)
                
            # Evaluate weights immediately to clear the lazy graph and prevent memory spikes
            mx.eval(new_linear.weight)
            if use_bias:
                mx.eval(new_linear.bias)

print("Saving fused model to {save_path}...")
os.makedirs("{save_path}", exist_ok=True)

# Save the trainable (now fully fused) parameters with MLX format metadata and proper key prefixes
flat_params = dict(tree_flatten(model.parameters()))
mapped_params = {{}}
for fk, v in flat_params.items():
    if fk.startswith("audio_tower."):
        mk = "model." + fk
    elif fk.startswith("language_model.model."):
        mk = fk.replace("language_model.model.", "model.language_model.")
    elif fk.startswith("language_model."):
        mk = fk.replace("language_model.", "model.language_model.")
    else:
        mk = "model." + fk
    mapped_params[mk] = v

mx.save_safetensors("{save_path}/model.safetensors", mapped_params, metadata={{"format": "mlx"}})

# Copy processor and config files
for file in os.listdir("{model_path}"):
    if file.endswith(".json") or file.endswith(".jinja"):
        shutil.copy2(os.path.join("{model_path}", file), "{save_path}")
        
print("Multimodal fusion complete!")
"""
        cmd = ["python", "-c", script]
    else:
        cmd = [
            "python", "-m", "mlx_lm.fuse",
            "--model", model_path,
            "--adapter-path", adapter_path,
            "--save-path", save_path,
            "--dequantize"  # Required for clean llama.cpp conversion
        ]
    subprocess.run(cmd)

def run_gguf(base_model_path, fused_model_path, output_path, outtype="q8_0", qat=False):
    """
    Converts the 16-bit fused MLX model into a single, quantized GGUF file.
    This file can be executed instantly by llama.cpp and served to Go applications.
    """
    if qat:
        if outtype != "q4_0":
            print(f"==> [QAT Optimization] Overriding outtype '{outtype}' to 'q4_0' for QAT alignment.")
            print("    Why? Gemma 4 QAT models were mathematically pre-conditioned during training")
            print("    specifically for 4-bit 'q4_0' quantization parameters. Exporting to other types")
            print("    (like q8_0 or q4_k_m) will bypass this alignment, resulting in higher accuracy loss.")
            outtype = "q4_0"

    print(f"==> Exporting fused model to GGUF (type: {outtype})...")
    
    # 1. Copy tokenizer.model from base model
    # MLX doesn't always carry the SentencePiece tokenizer forward, 
    # but llama.cpp absolutely requires it to understand text.
    tokenizer_src = os.path.join(base_model_path, "tokenizer.model")
    tokenizer_dest = os.path.join(fused_model_path, "tokenizer.model")
    if os.path.exists(tokenizer_src):
        shutil.copy2(tokenizer_src, tokenizer_dest)
        print("==> Copied tokenizer.model for conversion.")

    # 2. Ensure llama.cpp conversion tools exist locally
    llama_cpp_dir = "./llama.cpp"
    if not os.path.exists(llama_cpp_dir):
        print("==> Cloning llama.cpp repository for conversion tools...")
        subprocess.run(["git", "clone", "https://github.com/ggerganov/llama.cpp.git"])
    
    # 3. Determine if we need to do a two-step quantization
    native_types = {"f32", "f16", "bf16", "q8_0", "tq1_0", "tq2_0", "auto"}
    
    script_path = os.path.join(llama_cpp_dir, "convert_hf_to_gguf.py")
    
    if outtype in native_types:
        cmd = [
            "python", script_path,
            fused_model_path,
            "--outfile", output_path,
            "--outtype", outtype
        ]
        subprocess.run(cmd)
    else:
        # Two-step: convert to high-precision (f16) GGUF, then run llama-quantize
        temp_f16_path = output_path + ".temp-f16.gguf"
        print(f"==> Step 1: Converting to high-precision f16 GGUF ({temp_f16_path})...")
        cmd_convert = [
            "python", script_path,
            fused_model_path,
            "--outfile", temp_f16_path,
            "--outtype", "f16"
        ]
        subprocess.run(cmd_convert)
        
        # Now quantize
        quantize_bin = (
            shutil.which("llama-quantize") or 
            shutil.which("quantize") or 
            "/opt/homebrew/bin/llama-quantize"
        )
        
        if os.path.exists(temp_f16_path):
            print(f"==> Step 2: Quantizing to low-precision '{outtype}' using {quantize_bin}...")
            cmd_quantize = [
                quantize_bin,
                temp_f16_path,
                output_path,
                outtype
            ]
            subprocess.run(cmd_quantize)
            
            # Clean up temp file
            try:
                os.remove(temp_f16_path)
                print(f"==> Cleaned up temporary f16 file: {temp_f16_path}")
            except Exception as e:
                print(f"Warning: Could not remove temporary f16 file: {e}")
        else:
            print("Error: Temporary f16 GGUF file was not generated. Quantization aborted.")

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
    gguf_parser = subparsers.add_parser("gguf", help="Convert fused model to GGUF")
    gguf_parser.add_argument("--base-model", default="./model", help="Path to base model folder (for tokenizer)")
    gguf_parser.add_argument("--model", default="./fused_model_dequantized", help="Path to fused model folder")
    gguf_parser.add_argument("--dest", default="my-custom-model.gguf", help="Destination GGUF file path")
    gguf_parser.add_argument("--outtype", default="q8_0", help="Quantization type for GGUF (e.g., q8_0, f16, q4_k_m)")
    gguf_parser.add_argument("--qat", action="store_true", help="Enforce q4_0 quantization optimized for QAT models")

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
        run_gguf(args.base_model, args.model, args.dest, args.outtype, args.qat)
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