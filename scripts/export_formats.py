"""
Multi-format export: GGUF (llama.cpp), LiteRT-LM (.litertlm) and MLX.

Runs in the gemmma project environment (`uv run scripts/export_formats.py ...`).
GGUF and MLX go through the same mlx-tune code paths as `mlxtune gguf` / `mlxtune fuse`.
"""

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


def handle_gguf_export(base_model, adapter, output_gguf, outtype="q4_k_m", qat=False):
    """
    GGUF compilation via mlx-tune (same code path as `mlxtune gguf`):
    fuse + dequantize -> llama.cpp convert_hf_to_gguf.py -> llama-quantize.
    Tokenizer files come only from the base model (never copied in from elsewhere).
    """
    print_banner("Target: Compiling Fused GGUF Model")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from gemmmma.cli import run_gguf

    adapter_path = adapter if adapter and os.path.exists(adapter) else None
    if adapter and not adapter_path:
        print_warning(f"Adapter path '{adapter}' not found; exporting the base model only.")
    run_gguf(base_model, output_gguf, outtype, qat, adapter_path=adapter_path)
    print_success(f"GGUF compiled successfully at: {output_gguf}")


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
        print_error("The 'litert-torch' package is not installed in this environment.")
        print("LiteRT export is kept out of the default dependencies (it pulls a large torch stack).")
        print("Run it in an environment that has it, e.g. eldamo-tune's scripts/compile_litert.py,")
        print(f"or: {BLUE}uv run --with litert-torch scripts/export_formats.py litert ...{RESET}")
        print("Note: 12B models need >90GB RAM for LiteRT export (FRICTION_LOG FL-004).")
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
    Fuses the adapter into the base with mlx-tune and writes a standard MLX-LM
    directory (safetensors + config). Like `mlx_lm.fuse` without --dequantize, a
    quantized base stays quantized; an unquantized QAT base stays full precision
    (quantize it afterwards with `mlx_lm convert -q` if needed).
    """
    print_banner("Target: Compiling Standard MLX Format")
    from mlx_tune import FastLanguageModel

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    model, tokenizer = FastLanguageModel.from_pretrained(base_model)
    if adapter and os.path.exists(adapter):
        model.load_adapter(adapter)
    else:
        print_warning(f"Adapter path '{adapter}' not found; exporting the base model only.")
    model.save_pretrained_merged(output_dir, tokenizer, save_method="merged_4bit")
    print_success(f"MLX Model compiled successfully at: {output_dir}")
    print("This directory can be directly imported using `mlx_lm` in Python")
    print("or imported natively in Swift on iOS using the mlx-swift library!")


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
    parser_gguf.add_argument("--outtype", default="q4_k_m", help="GGUF quantization type (q4_k_m, q4_0, q8_0, f16, ...)")
    parser_gguf.add_argument("--qat", action="store_true", help="Enforce q4_0 for Gemma QAT checkpoints")

    # 2. LiteRT-LM Subparser
    parser_litert = subparsers.add_parser("litert", parents=[parent_parser], help="Export merged model to LiteRT-LM Flatbuffer")
    parser_litert.add_argument("--dest", default="./models/litert", help="Output folder to store .litertlm file")
    parser_litert.add_argument("--prefused", default="./fused_model_dequantized", help="Path to pre-fused, dequantized model weights (mlx fuse output)")

    # 3. MLX Subparser
    parser_mlx = subparsers.add_parser("mlx", parents=[parent_parser], help="Export fused model to standard MLX format folder")
    parser_mlx.add_argument("--dest", default="./models/mlx_model", help="Output MLX model folder path")

    args = parser.parse_args()

    if args.target == "gguf":
        handle_gguf_export(args.base, args.adapter, args.dest, args.outtype, args.qat)
    elif args.target == "litert":
        handle_litert_export(args.base, args.adapter, args.dest, args.prefused)
    elif args.target == "mlx":
        handle_mlx_export(args.base, args.adapter, args.dest)

if __name__ == "__main__":
    main()
