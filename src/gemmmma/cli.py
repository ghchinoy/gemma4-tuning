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

def run_train(model_path, data_path, iters, batch_size, log_file="training_log.jsonl", multimodal=False, tune_audio_encoder=False):
    """
    Executes the LoRA training loop on the Apple GPU (Metal).
    This freezes the base model and only updates a tiny set of adapter weights.
    Logs metrics to a JSONL file for external monitoring.
    """
    import re
    import datetime
    
    print(f"==> Starting LoRA training with MLX...")
    
    if multimodal:
        # Modern mlx_vlm CLI syntax
        cmd = [
            "python", "-m", "mlx_vlm.lora",
            "--model-path", model_path,
            "--dataset", data_path,
            "--iters", str(iters),
            "--batch-size", str(batch_size),
            "--output-path", "adapters"
        ]
        
        if tune_audio_encoder:
            print("==> Targeting Audio Encoder layers for Multimodal LoRA tuning.")
            # Adjust if mlx_vlm uses specific regex or flags for fine-tuning specific components
            # e.g., cmd.extend(["--fine-tune-type", "audio_encoder"]) 
    else:
        cmd = [
            "python", "-m", "mlx_lm.lora",
            "--model", model_path,
            "--train",
            "--data", data_path,
            "--iters", str(iters),
            "--batch-size", str(batch_size)
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
            "multimodal": multimodal
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

print("Saving fused model to {save_path}...")
os.makedirs("{save_path}", exist_ok=True)

# Save the trainable (now fully fused) parameters
mx.save_safetensors("{save_path}/model.safetensors", dict(tree_flatten(model.parameters())))

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

def run_gguf(base_model_path, fused_model_path, output_path, outtype="q8_0"):
    """
    Converts the 16-bit fused MLX model into a single, quantized GGUF file.
    This file can be executed instantly by llama.cpp and served to Go applications.
    """
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
    
    # 3. Execute the conversion script
    script_path = os.path.join(llama_cpp_dir, "convert_hf_to_gguf.py")
    cmd = [
        "python", script_path,
        fused_model_path,
        "--outfile", output_path,
        "--outtype", outtype
    ]
    subprocess.run(cmd)

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

    # 5. Eval Command
    eval_parser = subparsers.add_parser("eval", help="Test the fine-tuned model")
    eval_parser.add_argument("--model", default="./model", help="Path to base model")
    eval_parser.add_argument("--adapter", default="./adapters", help="Path to trained adapter directory")
    eval_parser.add_argument("--prompt", required=True, help="Text prompt to test")
    eval_parser.add_argument("--multimodal", action="store_true", help="Use mlx_vlm for multimodal evaluation")

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

    # 8. Clean Command
    clean_parser = subparsers.add_parser("clean", help="Clean up training artifacts to start fresh")

    args = parser.parse_args()

    if args.command == "download":
        download_model(args.model, args.dest)
    elif args.command == "prep":
        prep_dataset(args.dataset, args.dataset_config, args.dest, args.samples)
    elif args.command == "train":
        run_train(args.model, args.data, args.iters, args.batch_size, args.log_file, args.multimodal, args.tune_audio_encoder)
    elif args.command == "eval":
        run_eval(args.model, args.adapter, args.prompt, args.multimodal)
    elif args.command == "fuse":
        run_fuse(args.model, args.adapter, args.dest, args.multimodal)
    elif args.command == "gguf":
        run_gguf(args.base_model, args.model, args.dest, args.outtype)
    elif args.command == "clean":
        run_clean()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()