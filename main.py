import argparse
import yaml
import torch

from trainer import Trainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="Path to config.yaml")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    config["device"] = config.get(
        "device", "cuda" if torch.cuda.is_available() else "cpu"
    )

    trainer = Trainer(**config)
    trainer.train()


# Entry point
if __name__ == "__main__":
    main()
