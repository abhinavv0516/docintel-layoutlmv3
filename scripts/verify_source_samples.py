"""
Verify one source image for each target RVL-CDIP class.
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

    output_dir = Path("data/output/source_samples")
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 60)
    print("SOURCE DATASET SAMPLE VERIFICATION")
    print("=" * 60)

    dataset = load_dataset(
        DATASET_NAME,
        split="train",
        streaming=True,
    )

    found = set()

    for sample in dataset:

        label_id = sample["label"]

        if label_id not in TARGET_LABELS:
            continue

        class_name = TARGET_LABELS[label_id]

        if class_name in found:
            continue

        image = sample["image"]

        output_path = (
            output_dir
            / f"{class_name}.png"
        )

        image.save(output_path)

        print(
            f"{class_name:15} "
            f"label={label_id:2} "
            f"size={image.size} "
            f"-> {output_path}"
        )

        found.add(class_name)

        if len(found) == len(TARGET_LABELS):
            break

    print("\n" + "=" * 60)
    print("VERIFICATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()