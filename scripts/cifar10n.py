import argparse
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("upstream", type=Path, help="Local official cifar-10-100n directory")
    args = parser.parse_args()
    source = args.upstream.resolve()
    dest = Path(__file__).resolve().parents[1] / "data/cifar10n/cifar-10-100n-main"
    for relative in ["data/CIFAR-10_human_ordered.npy", "image_order_c10.npy"]:
        src = source / relative
        if not src.is_file():
            parser.error(f"Required official release file is missing: {src}")
    for relative in ["data/CIFAR-10_human_ordered.npy", "image_order_c10.npy"]:
        target = dest / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if target.read_bytes() != (source / relative).read_bytes():
                parser.error(f"Refusing to overwrite a different existing file: {target}")
        else:
            shutil.copy2(source / relative, target)
    print(f"Prepared labels and ordering metadata in {dest}")
    print("experiments.py verifies their alignment with torchvision images.")


if __name__ == "__main__":
    main()
