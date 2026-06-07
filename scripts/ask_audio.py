# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "mlx-vlm",
# ]
# ///

import sys
import argparse
from mlx_vlm import load, generate

def main():
    parser = argparse.ArgumentParser(description="Test Gemma 4 audio natively using mlx-vlm on Apple Silicon.")
    parser.add_argument("audio_path", help="Path to the .wav audio file")
    parser.add_argument("--prompt", default="Describe the speaker's tone in this audio and translate it to English.", help="Text prompt for the model")
    parser.add_argument("--model", default="model", help="Path to the MLX model directory (e.g., 'model' or 'mlx-community/gemma-4-E2B-it-4bit')")
    parser.add_argument("--max-tokens", type=int, default=256, help="Maximum number of tokens to generate")
    
    args = parser.parse_args()

    print(f"Loading MLX-VLM model from '{args.model}'...")
    try:
        model, processor = load(args.model)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Ensure you are pointing to a valid MLX model directory or Hugging Face repo.")
        sys.exit(1)

    print(f"Processing audio: {args.audio_path}")
    print(f"Prompt: {args.prompt}")

    # Build the message structure required for Gemma 4 multimodal inputs
    messages = [
        {"role": "user", "content": [
            {"type": "text", "text": args.prompt},
            {"type": "audio"} 
        ]}
    ]

    try:
        # Apply chat template
        formatted_prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    except Exception as e:
        print(f"Warning: Error formatting prompt via template ({e}). Falling back to simple formatting...")
        # Fallback if the processor doesn't natively handle the dictionary structure yet
        formatted_prompt = f"User: {args.prompt} <audio>\nAssistant:"

    print("\nGenerating response natively on Apple Silicon GPU...")
    
    try:
        # mlx_vlm.generate takes an audio parameter which is a list of file paths
        response = generate(
            model,
            processor,
            prompt=formatted_prompt,
            audio=[args.audio_path],
            max_tokens=args.max_tokens,
            verbose=False # Set to True if you want to see timing metrics from MLX
        )
        print("\n=== Model Response ===")
        print(response)
    except Exception as e:
        print(f"\nGeneration failed: {e}")
        print("This usually happens if the model directory is missing the audio projector weights,")
        print("or if the specific audio encoder architecture hasn't been mapped in your version of mlx-vlm.")

if __name__ == "__main__":
    main()
