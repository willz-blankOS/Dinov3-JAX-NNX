"""
    This script acts as a CLI tool when run from the terminal. It's o
    perations involve loading and saving DINOV3 models.
    
    Every model size available on huggingface is compatible, with the exception
    of the Convolution Based DINOv3 Models.
"""
import argparse

def load_model() -> None:
    ...

def load_and_save_model() -> None:
    ...

def load_cli_options() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    return parser

if __name__ == "__main__":
    ...