"""
Verify RVL-CDIP source labels against actual downloaded images.
"""

from pathlib import Path

from datasets import load_dataset


DATASET_NAME = "hf-tuner/rvl-cdip-document-classification"

TARGET_LABELS = {
    1: "form",
    4: "advertisement",
    10: "budget",
    11: "invoice",
    14: "resume",
}


def main():

    print("=" * 60)
    print("VERIFYING RVL-CDIP LABEL MAPPING")
    print("=" * 60)

    dataset = load_dataset(
        DATASET_NAME,
        split="train",
        streaming=True,
    )

    label_names = dataset.features["label"].names

    print("\nDataset labels:")

    for index, name in enumerate(label_names):
        print(f"{index}: {name}")

    print("\nTarget labels:")

    for label_id, name in TARGET_LABELS.items():
        print(f"{label_id}: {name}")

    print("\nChecking first sample from each target class...\n")

    found = set()

    for sample in dataset:

        label_id = sample["label"]

        if label_id not in TARGET_LABELS:
            continue

        class_name = TARGET_LABELS[label_id]

        if class_name in found:
            continue

        image = sample["image"]

        print("-" * 60)
        print(f"Label ID : {label_id}")
        print(f"Class    : {class_name}")
        print(f"Image    : {image.size}")
        print(f"Mode     : {image.mode}")

        output_path = Path(
            f"data/output/source_{class_name}.png"
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        image.save(output_path)

        print(f"Saved    : {output_path}")

        found.add(class_name)

        if len(found) == len(TARGET_LABELS):
            break

    print("\n" + "=" * 60)
    print("SOURCE LABEL VERIFICATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()