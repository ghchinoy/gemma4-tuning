# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "torch>=2.0.0",
#     "transformers>=4.38.0",
#     "peft>=0.9.0",
#     "mlx-lm>=0.10.0",
#     "huggingface-hub>=0.22.0",
#     "gguf>=0.18.0",
#     "safetensors>=0.4.0",
# ]
# ///

import os
import sys
import shutil
import argparse
import subprocess
from pathlib import Path

# Color and formatting constants for premium CLI logging
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BLUE = "\033[94m"
BOLD = "\033[1m"
RESET = "\033[0m"

def print_banner(text):
    print(f"\n{BLUE}{BOLD}{'=' * 60}{RESET}")
    print(f"{BLUE}{BOLD}  {text}{RESET}")
    print(f"{BLUE}{BOLD}{'=' * 60}{RESET}")

def print_success(text):
    print(f"{GREEN}{BOLD}✓ Success: {text}{RESET}")

def print_warning(text):
    print(f"{YELLOW}{BOLD}⚠ Warning: {text}{RESET}")

def print_error(text):
    print(f"{RED}{BOLD}✗ Error: {text}{RESET}")

def merge_lora_weights(base_model_path, adapter_path, merged_output_path):
    """
    Merges local LoRA adapter weights natively back into the base model weights
    using Hugging Face's transformers and PEFT libraries.
    This dequantizes/compiles weights into single standard Safetensors files,
    which is a prerequisite for LiteRT-LM conversion.
    """
    print_banner("Phase 1: Loading & Merging LoRA Weights into Base Model")
    print(f"Base model path: {base_model_path}")
    print(f"Adapter path: {adapter_path}")
    print(f"Saving merged weights to: {merged_output_path}")

    # Validate inputs
    if not os.path.exists(base_model_path):
        print_error(f"Base model path does not exist: {base_model_path}")
        sys.exit(1)
    if not os.path.exists(adapter_path):
        print_error(f"Adapter path does not exist: {adapter_path}")
        sys.exit(1)

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel

        print("\n==> Loading base model (this may take a few minutes)...")
        # Load the base model. Note: To merge safely for general inference, 
        # we load it in float16/bfloat16.
        torch_dtype = torch.bfloat16 if torch.cuda.is_available() or hasattr(torch, "backends") and torch.backends.mps.is_available() else torch.float32
        
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch_dtype,
            device_map="auto",
            low_cpu_mem_usage=True
        )
        tokenizer = AutoTokenizer.from_pretrained(base_model_path)

        print("==> Loading LoRA adapters...")
        model = PeftModel.from_pretrained(base_model, adapter_path)

        print("==> Mathematically fusing weights (merge and unload)...")
        merged_model = model.merge_and_unload()

        print(f"==> Writing merged model tensors to {merged_output_path}...")
        os.makedirs(merged_output_path, exist_ok=True)
        merged_model.save_pretrained(merged_output_path, safe_serialization=True)
        tokenizer.save_pretrained(merged_output_path)

        # Ensure tokenizer.model or standard vocabulary file is present
        vocab_file = Path(base_model_path) / "tokenizer.model"
        if vocab_file.exists():
            shutil.copy2(vocab_file, Path(merged_output_path) / "tokenizer.model")
            print("==> Copied tokenizer.model to merged directory.")

        print_success("LoRA weights merged successfully!")
        return True

    except Exception as e:
        print_error(f"Failed to merge LoRA weights: {e}")
        print("\nEnsure you are using standard PyTorch base model weights (not MLX-only 4bit weights)")
        print("for this step. LiteRT and GGUF conversions expect raw PyTorch weights.")
        sys.exit(1)


def handle_gguf_export(base_model, adapter, output_gguf, outtype="q4_k_m"):
    """
    Handles GGUF compilation. Re-uses mlx dequantization/fusion to construct GGUF.
    """
    print_banner("Target: Compiling Fused GGUF Model")
    temp_fused_dir = "./temp_fused_for_gguf"
    
    # 1. Fuse base model and adapter dequantizing to 16bit
    print("==> Step 1: Dequantizing and fusing weights via mlx_lm.fuse...")
    if os.path.exists(temp_fused_dir):
        shutil.rmtree(temp_fused_dir)
        
    cmd_fuse = [
        "python", "-m", "mlx_lm.fuse",
        "--model", base_model,
        "--adapter-path", adapter,
        "--save-path", temp_fused_dir,
        "--dequantize"
    ]
    
    try:
        subprocess.run(cmd_fuse, check=True)
        print_success("Dequantized fusion complete.")
    except subprocess.CalledProcessError as e:
        print_error(f"Dequantized fusion failed: {e}")
        sys.exit(1)

    # 2. Copy tokenizer.model from base model
    vocab_src = Path(base_model) / "tokenizer.model"
    vocab_dst = Path(temp_fused_dir) / "tokenizer.model"
    if vocab_src.exists():
        shutil.copy2(vocab_src, vocab_dst)
        print("==> Copied tokenizer.model for GGUF parsing.")

    # 3. Clone or resolve llama.cpp
    llama_cpp_dir = Path("./llama.cpp")
    if not llama_cpp_dir.exists():
        print("==> llama.cpp utility directory not found. Cloning helper tool repo...")
        try:
            subprocess.run(["git", "clone", "https://github.com/ggerganov/llama.cpp.git"], check=True)
        except Exception as e:
            print_error(f"Failed to clone llama.cpp: {e}")
            sys.exit(1)

    # 4. Run conversion script
    convert_script = llama_cpp_dir / "convert_hf_to_gguf.py"
    if not convert_script.exists():
        print_error(f"Could not find convert_hf_to_gguf.py script inside {llama_cpp_dir}")
        sys.exit(1)

    print("\n==> Step 2: Compiling Safetensors to GGUF format...")
    cmd_convert = [
        "python", str(convert_script),
        temp_fused_dir,
        "--outfile", output_gguf,
        "--outtype", outtype
    ]
    
    try:
        subprocess.run(cmd_convert, check=True)
        print_success(f"GGUF compiled successfully at: {output_gguf}")
    except subprocess.CalledProcessError as e:
        print_error(f"GGUF compilation script failed: {e}")
        sys.exit(1)
    finally:
        # Clean up temporary fused folder
        if os.path.exists(temp_fused_dir):
            shutil.rmtree(temp_fused_dir)


def handle_litert_export(base_model, adapter, output_dir, prefused_path=None):
    """
    Handles LiteRT-LM conversion. Uses litert-torch to compile standard
    HuggingFace weights into an optimized .litertlm mobile package.
    Supports leveraging pre-fused models from mlxtune to bypass heavy PEFT loading.
    """
    print_banner("Target: Compiling LiteRT-LM (.litertlm Model Package)")
    
    # Prerequisite check: can we import litert_torch / torch?
    litert_installed = False
    try:
        import litert_torch
        litert_installed = True
    except ImportError:
        pass

    temp_merged_dir = "./temp_merged_for_litert"
    source_model_dir = None

    # 1. Determine source model weights (try pre-fused first)
    if prefused_path and os.path.exists(prefused_path):
        print_success(f"Leveraging pre-fused, dequantized weights from '{prefused_path}'")
        print("This bypasses expensive PyTorch PEFT merging and saves gigabytes of RAM overhead!")
        source_model_dir = prefused_path
    else:
        print_warning(f"Pre-fused directory '{prefused_path}' not found. Falling back to on-the-fly PEFT merging...")
        # Fallback to merging on-the-fly via PyTorch PEFT if pre-fused is not available
        if os.path.exists(temp_merged_dir):
            shutil.rmtree(temp_merged_dir)
        
        merge_lora_weights(base_model, adapter, temp_merged_dir)
        source_model_dir = temp_merged_dir

    # 2. Check and run litert-torch compilation
    print("\n==> Step 2: Compiling Safetensors to LiteRT-LM format...")
    os.makedirs(output_dir, exist_ok=True)
    
    cmd_convert = [
        "litert-torch", "export_hf",
        f"--model={source_model_dir}",
        f"--output_dir={output_dir}",
        "--quantization_recipe=dynamic_wi8_afp32",  # Highly optimized dynamic 8-bit weight quantization
        "--externalize_embedder"
    ]

    if not litert_installed:
        print_warning("The 'litert-torch' package is not installed in the active environment.")
        print("To run the conversion, we will attempt to install it in this subprocess.")
        print("Alternatively, you can run: pip install litert-torch torch transformers")
        
        try:
            # We install litert-torch and other prerequisites
            subprocess.run([sys.executable, "-m", "pip", "install", "litert-torch", "torch", "transformers"], check=True)
            litert_installed = True
        except Exception as e:
            print_error(f"Failed to auto-install litert-torch: {e}")
            print("\nPlease install litert-torch manually and run the command again:")
            print(f"  {BLUE}pip install litert-torch torch transformers{RESET}")
            print(f"Then execute the compiled command directly:")
            print(f"  {BLUE}{' '.join(cmd_convert)}{RESET}")
            sys.exit(1)

    print(f"Running command: {' '.join(cmd_convert)}")
    try:
        subprocess.run(cmd_convert, check=True)
        print_success(f"LiteRT-LM model package compiled successfully!")
        print(f"Your mobile-ready model is located at: {output_dir}/model.litertlm")
    except subprocess.CalledProcessError as e:
        print_error(f"LiteRT-LM compilation failed: {e}")
        sys.exit(1)
    finally:
        # Clean up temporary merged folder if we created it
        if source_model_dir == temp_merged_dir and os.path.exists(temp_merged_dir):
            shutil.rmtree(temp_merged_dir)


def handle_mlx_export(base_model, adapter, output_dir):
    """
    Fuses the model and exports it as a clean, standard MLX-LM compatible
    directory with safetensors and config parameters.
    """
    print_banner("Target: Compiling Standard MLX Format")
    
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
        
    cmd_fuse = [
        "python", "-m", "mlx_lm.fuse",
        "--model", base_model,
        "--adapter-path", adapter,
        "--save-path", output_dir
    ]
    
    print("==> Compiling weights using mlx_lm.fuse...")
    try:
        subprocess.run(cmd_fuse, check=True)
        print_success(f"MLX Model compiled successfully at: {output_dir}")
        print("This directory can be directly imported using `mlx_lm` in Python")
        print("or imported natively in Swift on iOS using the mlx-swift library!")
    except subprocess.CalledProcessError as e:
        print_error(f"MLX compilation failed: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Mithlond Tuning Team Multi-Format Mobile & Desktop Model Compiler",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
  {BOLD}GGUF (Desktop & llama.cpp){RESET}:
    uv run scripts/export_formats.py gguf --base ./model --adapter ./adapters --dest my_model.gguf
    
  {BOLD}LiteRT-LM (iOS & Android Offline flatbuffers){RESET}:
    uv run scripts/export_formats.py litert --base ./model --adapter ./adapters --dest ./models/litert
    
  {BOLD}MLX (Apple Silicon Swift native){RESET}:
    uv run scripts/export_formats.py mlx --base ./model --adapter ./adapters --dest ./models/mlx_model
"""
    )
    
    subparsers = parser.add_subparsers(dest="target", required=True, help="Target format")
    
    # Shared parent parser for common arguments
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument("--base", default="./model", help="Path to base PyTorch model weights (unquantized)")
    parent_parser.add_argument("--adapter", default="./adapters", help="Path to fine-tuned LoRA adapter folder")

    # 1. GGUF Subparser
    parser_gguf = subparsers.add_parser("gguf", parents=[parent_parser], help="Export fused model to GGUF (llama.cpp)")
    parser_gguf.add_argument("--dest", default="my-custom-model.gguf", help="Output GGUF file path")
    parser_gguf.add_argument("--outtype", default="q4_k_m", choices=["q4_k_m", "q8_0", "f16"], help="GGUF Quantization type")

    # 2. LiteRT-LM Subparser
    parser_litert = subparsers.add_parser("litert", parents=[parent_parser], help="Export merged model to LiteRT-LM Flatbuffer")
    parser_litert.add_argument("--dest", default="./models/litert", help="Output folder to store .litertlm file")
    parser_litert.add_argument("--prefused", default="./fused_model_dequantized", help="Path to pre-fused, dequantized model weights (mlx fuse output)")

    # 3. MLX Subparser
    parser_mlx = subparsers.add_parser("mlx", parents=[parent_parser], help="Export fused model to standard MLX format folder")
    parser_mlx.add_argument("--dest", default="./models/mlx_model", help="Output MLX model folder path")

    args = parser.parse_args()

    if args.target == "gguf":
        handle_gguf_export(args.base, args.adapter, args.dest, args.outtype)
    elif args.target == "litert":
        handle_litert_export(args.base, args.adapter, args.dest, args.prefused)
    elif args.target == "mlx":
        handle_mlx_export(args.base, args.adapter, args.dest)

if __name__ == "__main__":
    main()
