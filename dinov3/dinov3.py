"""
    This script acts as a CLI tool when run from the terminal. It's o
    perations involve loading and saving DINOV3 models.
    
    Every model size available on huggingface is compatible, with the exception
    of the Convolution Based DINOv3 Models.
"""
import argparse

import jax
import torch
import flax.nnx as nnx
from transformers import AutoModel

import io.convert as convert

from rich.console import Console
from rich.style import Style

console = Console()

SUPPORTED_MODELS = [
    # LVD
    "dinov3-vit7b16-pretrain-lvd1689m",
    "dinov3-vit7h16plus-pretrain-lvd1689m",
    "dinov3-vitl16-pretrain-lvd1689m"
    "dinov3-vitb16-pretrain-lvd1689m",
    "dinov3-vits16plus-pretrain-lvd1689m"
    "dinov3-vits16-pretrain-lvd1689m",
    # SAT
    "dinov3-vit7b16-pretrain-sat494m",
    "dinov3-vitl16-pretrain-sat494m"
]

def load_model(model_name: str) -> None:
    torch_model = AutoModel.from_pretrained(
        "facebook/" + model_name,
    )

def load_and_save_model() -> None:
    ...

def load_cli_options() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "name", "-n", type=str, required=True, default="Name of DINOv3 model on HuggingFace"
    )
    return parser

if __name__ == "__main__":
    args = load_cli_options()

    # Check supported models
    model_name: str = args.name
    if model_name.lower() not in SUPPORTED_MODELS:
        console.print('\n This model is unsupported by this tool.\nSupported models: {SUPPORTED_MODELS}', style=Style(color="red"))

    ...