"""
Convert your training checkpoint into a local
Hugging Face-compatible folder with config + model weights.

Example:
  python convert_ckpt_to_hf.py \
    --ckpt_path <your model checkpoint path here> \
    --out_dir custom_model_hf

After conversion, you can load the tokenizer and your model with:
  from transformers import AutoTokenizer, AutoModelForCausalLM
  tok = AutoTokenizer.from_pretrained("tokenizer/Qwen3-0.6B-Base", trust_remote_code=True)
  model = AutoModelForCausalLM.from_pretrained("custom_model_hf", trust_remote_code=True)
"""

import argparse
import json
import os
import shutil
from pathlib import Path
import torch

from custom_model.configuration_custom_model import CustomModelConfig
from custom_model.modeling_custom_model import CustomModelForCausalLM


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_path", required=True, help="Path to .pt checkpoint from training")
    ap.add_argument("--out_dir", required=True, help="Output folder for HF model")
    ap.add_argument("--model_config", default=str(Path(__file__).parent / "src/settings/model_params.json"))
    return ap.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"[info] Loading training checkpoint from {args.ckpt_path}")
    ckpt = torch.load(args.ckpt_path, map_location="cpu")
    model_args = json.loads(Path(args.model_config).read_text())

    cfg = CustomModelConfig(**model_args)

    CustomModelConfig.register_for_auto_class()
    CustomModelForCausalLM.register_for_auto_class("AutoModelForCausalLM")

    print("[info] Instantiating HF model wrapper and loading weights...")
    cfg.tie_word_embeddings = True
    model = CustomModelForCausalLM(cfg)
    missing, unexpected = model.transformer.load_state_dict(ckpt["model_state"], strict=True)
    if missing:
        print("[warn] Missing keys:", missing)
    if unexpected:
        print("[warn] Unexpected keys:", unexpected)

    # Persist RoPE frequencies: HF's meta-device loading can leave this buffer uninitialized.
    rope = model.transformer.rotary_emb
    rope.register_buffer("inv_freq", rope.inv_freq, persistent=True)

    print(f"[info] Saving HF model to {args.out_dir}")
    model.save_pretrained(args.out_dir, safe_serialization=False)

    cfg.save_pretrained(args.out_dir)

    # Make the exported folder self-contained by copying the code package
    pkg_src = os.path.join(os.path.dirname(__file__), "custom_model")
    pkg_dst = os.path.join(args.out_dir, "custom_model")
    if not os.path.exists(pkg_dst):
        print(f"[info] Copying model code to {pkg_dst}")
        shutil.copytree(pkg_src, pkg_dst)

    # Match the exported model definition to the newly persistent buffer.
    for model_file in (Path(args.out_dir) / "model.py", Path(pkg_dst) / "model.py"):
        source = model_file.read_text()
        source = source.replace(
            'self.register_buffer("inv_freq", inv_freq, persistent=False)',
            'self.register_buffer("inv_freq", inv_freq, persistent=True)',
        )
        model_file.write_text(source)

    print("[done] Export complete. You can now load with AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True)")


if __name__ == "__main__":
    main()
