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

def prep_dataset(dataset_id, dest_dir, max_samples=None):
    """
    Downloads a dataset from Hugging Face and prepares it into the JSONL 
    ChatML format required by MLX for fine-tuning.
    """
    print(f"==> Downloading and preparing dataset '{dataset_id}'...")
    os.makedirs(dest_dir, exist_ok=True)
    
    try:
        ds = load_dataset(dataset_id, split="train")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("\nNOTE: GAIR/lima is a 'gated' dataset. To use it, you must:")
        print("  1. Go to https://huggingface.co/datasets/GAIR/lima and accept the terms.")
        print("  2. Run `huggingface-cli login` in your terminal.")
        print("For this tutorial, try running with the open dataset: --dataset yahma/alpaca-cleaned")
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
        prompt = row.get("instruction", row.get("prompt", ""))
        if row.get("input"): 
            prompt += "\n\nContext: " + row.get("input", "")
        
        completion = row.get("output", row.get("completion", row.get("response", "")))
        
        return {
            "messages": [
                {"role": "user", "content": prompt}, 
                {"role": "assistant", "content": completion}
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

def run_train(model_path, data_path, iters, batch_size, log_file="training_log.jsonl"):
    """
    Executes the LoRA training loop on the Apple GPU (Metal).
    This freezes the base model and only updates a tiny set of adapter weights.
    Logs metrics to a JSONL file for external monitoring.
    """
    import re
    import datetime
    
    print(f"==> Starting LoRA training with MLX...")
    cmd = [
        "python", "-m", "mlx_lm", "lora",
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
            "batch_size": batch_size
        }) + "\n")
        
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    
    train_pattern = re.compile(r"Iter\s+(\d+):\s+Train loss\s+([\d.]+),\s+Learning Rate\s+([\d.e+-]+),\s+It/sec\s+([\d.]+),\s+Tokens/sec\s+([\d.]+),\s+Trained Tokens\s+(\d+),\s+Peak mem\s+([\d.]+)\s+GB")
    val_pattern = re.compile(r"Iter\s+(\d+):\s+Val loss\s+([\d.]+),\s+Val took\s+([\d.]+)s")
    
    for line in process.stdout:
        print(line, end="") # Keep printing to the terminal
        
        log_entry = None
        train_match = train_pattern.search(line)
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
            
        val_match = val_pattern.search(line)
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

def run_eval(model_path, adapter_path, prompt):
    """
    Tests the newly trained adapter by loading the base model, 
    injecting the adapter weights in memory, and generating text.
    """
    print(f"==> Evaluating model...")
    cmd = [
        "python", "-m", "mlx_lm", "generate",
        "--model", model_path,
        "--adapter-path", adapter_path,
        "--prompt", prompt,
        "--max-tokens", "200"
    ]
    subprocess.run(cmd)

def run_fuse(model_path, adapter_path, save_path):
    """
    Permanently bakes the trained LoRA adapters into the base model.
    Crucially, it uses --dequantize to convert the Apple-specific 4-bit MLX 
    format back into standard 16-bit PyTorch tensors so llama.cpp can read it.
    """
    print(f"==> Fusing LoRA adapters into base model (dequantizing for GGUF compatibility)...")
    cmd = [
        "python", "-m", "mlx_lm", "fuse",
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
    prep_parser.add_argument("--dest", default="./data", help="Destination folder for JSONL files")
    prep_parser.add_argument("--samples", type=int, default=1000, help="Max samples to use for fast training")

    # 4. Train Command
    train_parser = subparsers.add_parser("train", help="Run LoRA training")
    train_parser.add_argument("--model", default="./model", help="Path to base model")
    train_parser.add_argument("--data", default="./data", help="Path to data folder")
    train_parser.add_argument("--iters", type=int, default=200, help="Number of training iterations (low for testing)")
    train_parser.add_argument("--batch-size", type=int, default=2, help="Batch size")
    train_parser.add_argument("--log-file", default="training_log.jsonl", help="File to output JSONL metrics")

    # 5. Eval Command
    eval_parser = subparsers.add_parser("eval", help="Test the fine-tuned model")
    eval_parser.add_argument("--model", default="./model", help="Path to base model")
    eval_parser.add_argument("--adapter", default="./adapters", help="Path to trained adapter directory")
    eval_parser.add_argument("--prompt", required=True, help="Text prompt to test")

    # 6. Fuse Command
    fuse_parser = subparsers.add_parser("fuse", help="Fuse adapters into base model")
    fuse_parser.add_argument("--model", default="./model", help="Path to base model")
    fuse_parser.add_argument("--adapter", default="./adapters", help="Path to trained adapter directory")
    fuse_parser.add_argument("--dest", default="./fused_model_dequantized", help="Destination folder for fused model")

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
        prep_dataset(args.dataset, args.dest, args.samples)
    elif args.command == "train":
        run_train(args.model, args.data, args.iters, args.batch_size, args.log_file)
    elif args.command == "eval":
        run_eval(args.model, args.adapter, args.prompt)
    elif args.command == "fuse":
        run_fuse(args.model, args.adapter, args.dest)
    elif args.command == "gguf":
        run_gguf(args.base_model, args.model, args.dest, args.outtype)
    elif args.command == "clean":
        run_clean()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()