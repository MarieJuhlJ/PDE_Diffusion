from pde_diff.loss import DarcyLoss
import torch.nn as nn
import torch
from lightning.pytorch.callbacks import Callback
from pde_diff.utils import CallbackRegistry

class SaveBestModel(Callback):
    def __init__(self):
        super().__init__()

    def on_validation_epoch_end(self, trainer, pl_module):
        if not pl_module.save_model or pl_module.trainer.global_rank != 0:
            return

        metric = pl_module.trainer.callback_metrics.get(pl_module.monitor)
        if metric is None:
            return

        current = float(metric.detach().cpu())

        if (
            pl_module.best_score is None
            or (pl_module.mode == "min" and current < pl_module.best_score)
            or (pl_module.mode == "max" and current > pl_module.best_score)
        ):
            pl_module.best_score = current

            tag = f"best-{pl_module.monitor}"
            ckpt_path = pl_module.save_dir / f"{tag}.ckpt"
            pl_module.trainer.save_checkpoint(ckpt_path)
            torch.save(pl_module.state_dict(), pl_module.save_dir / f"{tag}-weights.pt")
            pl_module._save_cfg(pl_module.save_dir)

            pl_module.log("best_model_improved", 1.0, prog_bar=True)

@CallbackRegistry.register("darcy")
class DarcyLogger(Callback):
    def __init__(self):
        super().__init__()
        self.darcy_loss = DarcyLoss(None)

    def on_validation_epoch_end(self, trainer, pl_module):
        pl_module.eval()
        with torch.no_grad():
            x0_preds = pl_module.sample_loop(batch_size=16)
            darcy_res = self.darcy_loss.darcy_residual_loss(x0_preds, 1.0)
        pl_module.log("val_darcy_residual", darcy_res, prog_bar=True, on_epoch=True, sync_dist=True)