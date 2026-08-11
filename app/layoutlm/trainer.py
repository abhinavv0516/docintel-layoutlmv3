"""
LayoutLMv3 Training Engine.

Handles:
    - model training
    - gradient accumulation
    - mixed precision
    - learning-rate scheduling
    - validation
    - checkpointing
    - training history
"""

import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from transformers import LayoutLMv3ForSequenceClassification

from app.layoutlm.cached_dataset import CachedDocumentDataset
from app.layoutlm.training import TrainingConfig


class DocumentTrainer:
    """Trainer for LayoutLMv3 document classification."""

    def __init__(self, config: TrainingConfig):
        self.config = config

        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        self._set_seed(config.seed)

        print(f"Training device: {self.device}")

        if self.device.type == "cuda":
            print(
                f"GPU: {torch.cuda.get_device_name(0)}"
            )

        # --------------------------------------------------
        # Model
        # --------------------------------------------------

        print("\nLoading LayoutLMv3 classifier...")

        self.model = (
            LayoutLMv3ForSequenceClassification
            .from_pretrained(
                config.model_name,
                num_labels=config.num_labels,
            )
        )

        self.model.to(self.device)

        # --------------------------------------------------
        # Datasets
        # --------------------------------------------------

        print("\nLoading training dataset...")

        self.train_dataset = CachedDocumentDataset(
            config.train_dir
        )

        print("Loading validation dataset...")

        self.validation_dataset = CachedDocumentDataset(
            config.validation_dir
        )

        # --------------------------------------------------
        # DataLoaders
        # --------------------------------------------------

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
        )

        self.validation_loader = DataLoader(
            self.validation_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
        )

        # --------------------------------------------------
        # Optimizer
        # --------------------------------------------------

        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # --------------------------------------------------
        # Scheduler
        # --------------------------------------------------

        updates_per_epoch = max(
            1,
            (
                len(self.train_loader)
                + config.gradient_accumulation_steps
                - 1
            )
            // config.gradient_accumulation_steps,
        )

        total_training_steps = (
            updates_per_epoch
            * config.epochs
        )

        warmup_steps = max(
            1,
            int(0.1 * total_training_steps),
        )

        self.total_training_steps = (
            total_training_steps
        )

        self.warmup_steps = warmup_steps

        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(
                    current_step + 1
                ) / float(warmup_steps)

            remaining_steps = (
                total_training_steps
                - current_step
            )

            decay_steps = max(
                1,
                total_training_steps
                - warmup_steps,
            )

            return max(
                0.0,
                float(remaining_steps)
                / float(decay_steps),
            )

        self.scheduler = LambdaLR(
            self.optimizer,
            lr_lambda,
        )

        print(
            f"\nTraining steps: "
            f"{total_training_steps}"
        )

        print(
            f"Warmup steps: "
            f"{warmup_steps}"
        )

        # --------------------------------------------------
        # Mixed precision
        # --------------------------------------------------

        self.use_amp = (
            self.device.type == "cuda"
        )

        self.scaler = torch.amp.GradScaler(
            "cuda",
            enabled=self.use_amp,
        )

        # --------------------------------------------------
        # Checkpoints
        # --------------------------------------------------

        self.checkpoint_dir = Path(
            config.checkpoint_dir
        )

        self.best_model_path = Path(
            config.best_model_path
        )

        self.checkpoint_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.best_model_path.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.best_accuracy = 0.0

        # --------------------------------------------------
        # Training history
        # --------------------------------------------------

        self.history = []

        self.history_path = (
            self.checkpoint_dir
            / "training_history.json"
        )

    @staticmethod
    def _set_seed(seed):
        """Set random seeds."""

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _move_batch_to_device(self, batch):
        """Move tensor batch to device."""

        return {
            key: value.to(
                self.device,
                non_blocking=True,
            )
            if torch.is_tensor(value)
            else value
            for key, value in batch.items()
        }

    def train_one_epoch(self, epoch):
        """Train for one epoch."""

        self.model.train()

        total_loss = 0.0
        correct = 0
        total = 0

        self.optimizer.zero_grad(
            set_to_none=True
        )

        accumulation_steps = (
            self.config.gradient_accumulation_steps
        )

        num_batches = len(
            self.train_loader
        )

        for step, batch in enumerate(
            self.train_loader
        ):

            batch = self._move_batch_to_device(
                batch
            )

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=self.use_amp,
            ):

                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch[
                        "attention_mask"
                    ],
                    bbox=batch["bbox"],
                    pixel_values=batch[
                        "pixel_values"
                    ],
                    labels=batch["labels"],
                )

                loss = (
                    outputs.loss
                    / accumulation_steps
                )

            self.scaler.scale(
                loss
            ).backward()

            is_last_batch = (
                step + 1 == num_batches
            )

            should_update = (
                (step + 1)
                % accumulation_steps
                == 0
                or is_last_batch
            )

            if should_update:

                self.scaler.step(
                    self.optimizer
                )

                self.scaler.update()

                self.scheduler.step()

                self.optimizer.zero_grad(
                    set_to_none=True
                )

            batch_loss = (
                outputs.loss.detach()
            )

            batch_size = (
                batch["labels"].size(0)
            )

            total_loss += (
                batch_loss.item()
                * batch_size
            )

            predictions = (
                outputs.logits.argmax(
                    dim=-1
                )
            )

            correct += (
                (
                    predictions
                    == batch["labels"]
                )
                .sum()
                .item()
            )

            total += batch_size

            if (
                (step + 1) % 100 == 0
                or step == 0
                or is_last_batch
            ):

                current_lr = (
                    self.optimizer.param_groups[
                        0
                    ]["lr"]
                )

                print(
                    f"Epoch {epoch} | "
                    f"Step {step + 1}/"
                    f"{num_batches} | "
                    f"Loss: "
                    f"{batch_loss.item():.4f} | "
                    f"LR: "
                    f"{current_lr:.2e}"
                )

        average_loss = (
            total_loss / total
        )

        accuracy = (
            correct / total
        )

        return average_loss, accuracy

    @torch.no_grad()
    def validate(self):
        """Evaluate model on validation set."""

        self.model.eval()

        total_loss = 0.0
        correct = 0
        total = 0

        for batch in self.validation_loader:

            batch = self._move_batch_to_device(
                batch
            )

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=self.use_amp,
            ):

                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch[
                        "attention_mask"
                    ],
                    bbox=batch["bbox"],
                    pixel_values=batch[
                        "pixel_values"
                    ],
                    labels=batch["labels"],
                )

            batch_size = (
                batch["labels"].size(0)
            )

            total_loss += (
                outputs.loss.item()
                * batch_size
            )

            predictions = (
                outputs.logits.argmax(
                    dim=-1
                )
            )

            correct += (
                (
                    predictions
                    == batch["labels"]
                )
                .sum()
                .item()
            )

            total += batch_size

        average_loss = (
            total_loss / total
        )

        accuracy = (
            correct / total
        )

        return average_loss, accuracy

    def save_best_model(self, accuracy):
        """Save best validation model."""

        if accuracy <= self.best_accuracy:
            return False

        self.best_accuracy = accuracy

        self.model.save_pretrained(
            self.best_model_path
        )

        print("\nNew best model saved!")

        print(
            f"Validation accuracy: "
            f"{accuracy:.4f}"
        )

        print(
            f"Path: "
            f"{self.best_model_path}"
        )

        return True

    def save_history(self):
        """Save training history to JSON."""

        with open(
            self.history_path,
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                self.history,
                file,
                indent=4,
            )

    def train(self):
        """Run complete training."""

        print(
            "\n" + "=" * 60
        )

        print("STARTING TRAINING")

        print(
            "=" * 60
        )

        for epoch in range(
            1,
            self.config.epochs + 1,
        ):

            print(
                "\n" + "=" * 60
            )

            print(
                f"EPOCH "
                f"{epoch}/"
                f"{self.config.epochs}"
            )

            print(
                "=" * 60
            )

            train_loss, train_accuracy = (
                self.train_one_epoch(
                    epoch
                )
            )

            print(
                "\nTraining results:"
            )

            print(
                f"Loss: "
                f"{train_loss:.4f}"
            )

            print(
                f"Accuracy: "
                f"{train_accuracy:.4f}"
            )

            print(
                "\nRunning validation..."
            )

            val_loss, val_accuracy = (
                self.validate()
            )

            print(
                "\nValidation results:"
            )

            print(
                f"Loss: "
                f"{val_loss:.4f}"
            )

            print(
                f"Accuracy: "
                f"{val_accuracy:.4f}"
            )

            current_lr = (
                self.optimizer.param_groups[
                    0
                ]["lr"]
            )

            self.history.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_accuracy": train_accuracy,
                    "validation_loss": val_loss,
                    "validation_accuracy": val_accuracy,
                    "learning_rate": current_lr,
                }
            )

            self.save_history()

            self.save_best_model(
                val_accuracy
            )

        print(
            "\n" + "=" * 60
        )

        print("TRAINING COMPLETE")

        print(
            "=" * 60
        )

        print(
            f"Best validation accuracy: "
            f"{self.best_accuracy:.4f}"
        )

        print(
            f"History saved to: "
            f"{self.history_path}"
        )