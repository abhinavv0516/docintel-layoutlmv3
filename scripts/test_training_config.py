"""
Test LayoutLMv3 training configuration.
"""

import sys
from pathlib import Path

sys.path.append(
    str(Path(__file__).resolve().parent.parent)
)

from app.layoutlm.training import (
    get_training_config,
)


def main():

    print("=" * 60)
    print("TRAINING CONFIGURATION")
    print("=" * 60)

    config = get_training_config()

    for key, value in vars(config).items():

        print(
            f"{key:35}: {value}"
        )

    print("\n" + "=" * 60)
    print("TRAINING CONFIG TEST PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()