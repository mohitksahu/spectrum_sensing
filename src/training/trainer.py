"""Training utilities for SpectraSense."""

from typing import Optional, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..utils.logger import ExperimentLogger
from ..utils.checkpoint import CheckpointManager
from ..losses.focal_loss import FocalLoss
from ..losses.uncertainty_loss import KendallUncertaintyLoss
from .callbacks import EarlyStopping, get_warmup_cosine_scheduler
from .metrics import compute_pu_metrics, compute_mod_metrics, compute_snr_metrics


class Trainer:
    """Lightweight trainer to orchestrate training loops.

    This is intentionally minimal: supports one epoch loops with AMP.
    """
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: str = "cpu",
        logger: Optional[ExperimentLogger] = None,
        checkpoint_manager: Optional[CheckpointManager] = None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.logger = logger
        self.checkpoint_manager = checkpoint_manager
        self.scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None

    def train_epoch(self, dataloader: DataLoader, loss_fn, epoch: int) -> Dict[str, float]:
        self.model.train()
        running_loss = 0.0
        n = 0
        for batch in dataloader:
            inputs, pu, mod, snr = batch
            inputs = inputs.to(self.device)
            pu = pu.to(self.device)
            mod = mod.to(self.device)
            snr = snr.to(self.device)

            self.optimizer.zero_grad()
            if self.scaler:
                with torch.cuda.amp.autocast():
                    outputs = self.model(inputs)
                    losses = loss_fn(outputs, {"pu": pu, "mod": mod, "snr": snr})
                    loss = losses["total"]
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(inputs)
                losses = loss_fn(outputs, {"pu": pu, "mod": mod, "snr": snr})
                loss = losses["total"]
                loss.backward()
                self.optimizer.step()

            running_loss += float(loss.item())
            n += 1

        avg_loss = running_loss / max(1, n)
        if self.logger:
            self.logger.info(f"Epoch {epoch}: train_loss={avg_loss:.4f}")
        return {"train_loss": avg_loss}

    def validate(self, dataloader: DataLoader, loss_fn) -> Dict[str, float]:
        self.model.eval()
        running_loss = 0.0
        n = 0
        with torch.no_grad():
            for batch in dataloader:
                inputs, pu, mod, snr = batch
                inputs = inputs.to(self.device)
                pu = pu.to(self.device)
                mod = mod.to(self.device)
                snr = snr.to(self.device)

                outputs = self.model(inputs)
                losses = loss_fn(outputs, {"pu": pu, "mod": mod, "snr": snr})
                running_loss += float(losses["total"].item())
                n += 1

        avg_loss = running_loss / max(1, n)
        if self.logger:
            self.logger.info(f"Validation: loss={avg_loss:.4f}")
        return {"val_loss": avg_loss}

    def fit(self, train_loader: DataLoader, val_loader: DataLoader, loss_fn, epochs: int = 1):
        for epoch in range(1, epochs + 1):
            train_metrics = self.train_epoch(train_loader, loss_fn, epoch)
            val_metrics = self.validate(val_loader, loss_fn)
            metrics = {**train_metrics, **val_metrics}
            if self.checkpoint_manager:
                self.checkpoint_manager.save(self.model, self.optimizer, None, epoch, metrics)


class SpectraSenseTrainer:
    """Trainer wrapper used by train.py for two-phase pipeline compatibility.

    This implementation performs supervised multi-task training and keeps
    interfaces compatible with the expected script flow.
    """

    def __init__(
        self,
        model: nn.Module,
        config: Dict[str, Any],
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        logger: Optional[ExperimentLogger] = None,
        pu_class_weights: Optional[torch.Tensor] = None,
        mod_class_weights: Optional[torch.Tensor] = None,
    ):
        self.model = model.to(device)
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.logger = logger

        self.pu_class_weights = pu_class_weights.to(device) if pu_class_weights is not None else None
        self.mod_class_weights = mod_class_weights.to(device) if mod_class_weights is not None else None

        phase2_cfg = config["phase2"]
        self.epochs = int(phase2_cfg.get("epochs", 1))
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(phase2_cfg.get("learning_rate", 1e-4)),
            weight_decay=float(phase2_cfg.get("weight_decay", 0.0)),
        )

        ckpt_cfg = config.get("checkpoint", {})
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=ckpt_cfg.get("dir", "checkpoints"),
            monitor=ckpt_cfg.get("monitor", "val_loss"),
            mode="min",
            save_best=ckpt_cfg.get("save_best", True),
            save_last=ckpt_cfg.get("save_last", True),
        )

        early_cfg = phase2_cfg.get("early_stopping", {})
        self.early_stopping = None
        if early_cfg.get("enabled", False):
            self.early_stopping = EarlyStopping(
                patience=int(early_cfg.get("patience", 8)),
                mode=str(early_cfg.get("mode", "min")),
            )

        self.pu_loss = FocalLoss(
            gamma=float(config.get("losses", {}).get("pu", {}).get("gamma", 2.0)),
            alpha=self.pu_class_weights,
        )
        # Phase 2 Decision 3: add modest label smoothing to the modulation head (previously 0.0)
        # to reduce overconfidence and narrow the test/train modulation gap.
        self.mod_loss = nn.CrossEntropyLoss(
            weight=self.mod_class_weights,
            label_smoothing=float(config.get("losses", {}).get("mod", {}).get("label_smoothing", 0.05)),
        )
        self.snr_loss = nn.HuberLoss(delta=float(config.get("losses", {}).get("snr", {}).get("delta", 1.0)))
        self.uncertainty = KendallUncertaintyLoss(num_tasks=3).to(device)

        # Train uncertainty parameters together with model parameters.
        self.optimizer.add_param_group({"params": self.uncertainty.parameters()})

        # Phase 2 Decision 4: use cosine annealing with warmup instead of the previous constant learning rate,
        # because the validation curve plateaued for multiple epochs before early stopping.
        self.scheduler = get_warmup_cosine_scheduler(
            self.optimizer,
            warmup_steps=max(1, int(0.05 * self.epochs)),
            total_steps=max(1, self.epochs),
            min_lr=0.1,
        )

        self.best_metrics: Dict[str, float] = {"val_loss": float("inf")}

    def _apply_snr_augmentation(self, x: torch.Tensor, snr: torch.Tensor) -> torch.Tensor:
        """Apply light training-only noise injection that is stronger for lower SNR samples."""
        snr = snr.view(-1, 1, 1).float()
        low_snr_scale = torch.clamp((10.0 - snr) / 10.0, min=0.0, max=1.0)
        noise_std = 0.01 + 0.04 * low_snr_scale

        # Phase 2 Decision 5: inject SNR-aware noise during training (previously no input augmentation)
        # so 4-6 dB examples are seen with slightly more corruption than cleaner bins.
        if x.dim() == 2:
            return x + torch.randn_like(x) * noise_std.squeeze(-1)
        return x + torch.randn_like(x) * noise_std

    def _compute_losses(self, outputs: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        raw_losses = {
            "pu": self.pu_loss(outputs["pu"], targets["pu"]),
            "mod": self.mod_loss(outputs["mod"], targets["mod"]),
            "snr": self.snr_loss(outputs["snr"].view(-1), targets["snr"].view(-1)),
        }
        weighted = self.uncertainty(raw_losses)
        weighted["pu_raw"] = raw_losses["pu"]
        weighted["mod_raw"] = raw_losses["mod"]
        weighted["snr_raw"] = raw_losses["snr"]
        return weighted

    def _run_epoch(self, training: bool) -> Dict[str, float]:
        loader = self.train_loader if training else self.val_loader
        self.model.train(mode=training)
        self.uncertainty.train(mode=training)

        total_loss = 0.0
        total_pu_raw = 0.0
        total_mod_raw = 0.0
        total_snr_raw = 0.0
        total_pu_weighted = 0.0
        total_mod_weighted = 0.0
        total_snr_weighted = 0.0
        num_batches = 0
        all_pu_logits = []
        all_mod_logits = []
        all_snr_preds = []
        all_pu_targets = []
        all_mod_targets = []
        all_snr_targets = []

        for x, y_pu, y_mod, y_snr in loader:
            x = x.to(self.device)
            y_pu = y_pu.to(self.device)
            y_mod = y_mod.to(self.device)
            y_snr = y_snr.to(self.device)

            if training:
                x = self._apply_snr_augmentation(x, y_snr)

            if training:
                self.optimizer.zero_grad(set_to_none=True)

            with torch.set_grad_enabled(training):
                outputs = self.model(x)
                weighted = self._compute_losses(outputs, {"pu": y_pu, "mod": y_mod, "snr": y_snr})
                loss = weighted["total"]
                if training:
                    loss.backward()
                    self.optimizer.step()

            all_pu_logits.append(outputs["pu"].detach().float().cpu())
            all_mod_logits.append(outputs["mod"].detach().float().cpu())
            all_snr_preds.append(outputs["snr"].detach().float().view(-1).cpu())
            all_pu_targets.append(y_pu.detach().cpu())
            all_mod_targets.append(y_mod.detach().cpu())
            all_snr_targets.append(y_snr.detach().float().view(-1).cpu())

            total_loss += float(loss.item())
            total_pu_raw += float(weighted["pu_raw"].item())
            total_mod_raw += float(weighted["mod_raw"].item())
            total_snr_raw += float(weighted["snr_raw"].item())
            total_pu_weighted += float(weighted["pu_weighted"].item())
            total_mod_weighted += float(weighted["mod_weighted"].item())
            total_snr_weighted += float(weighted["snr_weighted"].item())
            num_batches += 1

        avg = total_loss / max(1, num_batches)
        pu_logits = torch.cat(all_pu_logits, dim=0) if all_pu_logits else torch.empty(0, 2)
        mod_logits = torch.cat(all_mod_logits, dim=0) if all_mod_logits else torch.empty(0, 5)
        snr_preds = torch.cat(all_snr_preds, dim=0) if all_snr_preds else torch.empty(0)
        pu_targets = torch.cat(all_pu_targets, dim=0) if all_pu_targets else torch.empty(0, dtype=torch.long)
        mod_targets = torch.cat(all_mod_targets, dim=0) if all_mod_targets else torch.empty(0, dtype=torch.long)
        snr_targets = torch.cat(all_snr_targets, dim=0) if all_snr_targets else torch.empty(0)

        epoch_metrics: Dict[str, float] = {}
        if pu_logits.numel() > 0:
            epoch_metrics.update(compute_pu_metrics(pu_logits, pu_targets))
        if mod_logits.numel() > 0:
            epoch_metrics.update(compute_mod_metrics(mod_logits, mod_targets))
        if snr_preds.numel() > 0:
            epoch_metrics.update(compute_snr_metrics(snr_preds, snr_targets))

        return {
            "loss": avg,
            "pu_raw": total_pu_raw / max(1, num_batches),
            "mod_raw": total_mod_raw / max(1, num_batches),
            "snr_raw": total_snr_raw / max(1, num_batches),
            "pu_weighted": total_pu_weighted / max(1, num_batches),
            "mod_weighted": total_mod_weighted / max(1, num_batches),
            "snr_weighted": total_snr_weighted / max(1, num_batches),
            "uncertainty_weights": self.uncertainty.log_vars.detach().cpu().tolist(),
            **epoch_metrics,
        }

    def train(self) -> Dict[str, Any]:
        history = {"train_loss": [], "val_loss": []}
        stopped_early = False
        best_checkpoint_path = self.checkpoint_manager.checkpoint_dir / "best.pt"

        for epoch in range(1, self.epochs + 1):
            train_out = self._run_epoch(training=True)
            val_out = self._run_epoch(training=False)

            train_loss = train_out["loss"]
            val_loss = val_out["loss"]
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history.setdefault("train_pu_raw", []).append(train_out["pu_raw"])
            history.setdefault("train_mod_raw", []).append(train_out["mod_raw"])
            history.setdefault("train_snr_raw", []).append(train_out["snr_raw"])
            history.setdefault("train_pu_weighted", []).append(train_out["pu_weighted"])
            history.setdefault("train_mod_weighted", []).append(train_out["mod_weighted"])
            history.setdefault("train_snr_weighted", []).append(train_out["snr_weighted"])
            history.setdefault("uncertainty_log_vars", []).append(train_out["uncertainty_weights"])
            history.setdefault("train_pu_acc", []).append(train_out.get("pu_acc"))
            history.setdefault("train_pu_f1", []).append(train_out.get("pu_f1"))
            history.setdefault("train_pu_auc", []).append(train_out.get("pu_auc"))
            history.setdefault("train_mod_acc", []).append(train_out.get("mod_acc"))
            history.setdefault("train_mod_f1", []).append(train_out.get("mod_f1"))
            history.setdefault("train_snr_mae", []).append(train_out.get("snr_mae"))
            history.setdefault("train_snr_rmse", []).append(train_out.get("snr_rmse"))
            history.setdefault("val_pu_acc", []).append(val_out.get("pu_acc"))
            history.setdefault("val_pu_f1", []).append(val_out.get("pu_f1"))
            history.setdefault("val_pu_auc", []).append(val_out.get("pu_auc"))
            history.setdefault("val_mod_acc", []).append(val_out.get("mod_acc"))
            history.setdefault("val_mod_f1", []).append(val_out.get("mod_f1"))
            history.setdefault("val_snr_mae", []).append(val_out.get("snr_mae"))
            history.setdefault("val_snr_rmse", []).append(val_out.get("snr_rmse"))

            metrics = {"train_loss": train_loss, "val_loss": val_loss}
            self.checkpoint_manager.save(
                model=self.model,
                optimizer=self.optimizer,
                scheduler=None,
                epoch=epoch,
                metrics=metrics,
                extra={"uncertainty_state_dict": self.uncertainty.state_dict()},
            )

            if self.scheduler is not None:
                self.scheduler.step()

            if val_loss < self.best_metrics["val_loss"]:
                self.best_metrics = {"val_loss": val_loss, "epoch": epoch}

            if self.logger:
                self.logger.info(
                    "Epoch %d/%d - train_loss=%.4f, val_loss=%.4f, "
                    "train(pu_acc=%.4f, pu_f1=%.4f, pu_auc=%.4f, mod_acc=%.4f, mod_f1=%.4f, snr_mae=%.4f, snr_rmse=%.4f), "
                    "val(pu_acc=%.4f, pu_f1=%.4f, pu_auc=%.4f, mod_acc=%.4f, mod_f1=%.4f, snr_mae=%.4f, snr_rmse=%.4f), "
                    "raw(pu=%.4f, mod=%.4f, snr=%.4f), "
                    "weighted(pu=%.4f, mod=%.4f, snr=%.4f), log_vars=%s"
                    % (
                        epoch,
                        self.epochs,
                        train_loss,
                        val_loss,
                        train_out.get("pu_acc", float("nan")),
                        train_out.get("pu_f1", float("nan")),
                        train_out.get("pu_auc", float("nan")),
                        train_out.get("mod_acc", float("nan")),
                        train_out.get("mod_f1", float("nan")),
                        train_out.get("snr_mae", float("nan")),
                        train_out.get("snr_rmse", float("nan")),
                        val_out.get("pu_acc", float("nan")),
                        val_out.get("pu_f1", float("nan")),
                        val_out.get("pu_auc", float("nan")),
                        val_out.get("mod_acc", float("nan")),
                        val_out.get("mod_f1", float("nan")),
                        val_out.get("snr_mae", float("nan")),
                        val_out.get("snr_rmse", float("nan")),
                        train_out["pu_raw"],
                        train_out["mod_raw"],
                        train_out["snr_raw"],
                        train_out["pu_weighted"],
                        train_out["mod_weighted"],
                        train_out["snr_weighted"],
                        [round(v, 4) for v in train_out["uncertainty_weights"]],
                    )
                )

            if self.early_stopping is not None and self.early_stopping.step(val_loss):
                stopped_early = True
                if self.logger:
                    self.logger.info(
                        f"Early stopping triggered at epoch {epoch} after {self.early_stopping.num_bad_epochs} bad epochs"
                    )
                break

        if best_checkpoint_path.exists():
            checkpoint = torch.load(best_checkpoint_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if self.logger:
                self.logger.info(f"Restored best checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")

        history["stopped_early"] = stopped_early
        history["best_epoch"] = self.best_metrics.get("epoch")
        return {"phase2": {"history": history}}
