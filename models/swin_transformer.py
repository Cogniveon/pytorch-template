import torch
import lightning.pytorch as pl
import torch.nn as nn

from torchvision.models import swin_v2_b, Swin_V2_B_Weights


class SwinTransformer(pl.LightningModule):
    def __init__(self, num_classes: int = 10, learning_rate: float = 1e-3):
        super().__init__()
        model = swin_v2_b(weights=Swin_V2_B_Weights.IMAGENET1K_V1)
        model.head = nn.Linear(model.head.in_features, num_classes)

        self.model = model

        self.example_input_array = torch.Tensor(1, 3, 224, 224)
        self.learning_rate = learning_rate

        self.save_hyperparameters()

    def forward(self, *args: torch.Any, **kwargs: torch.Any) -> torch.Any:
        return self.model(*args, **kwargs)

    def training_step(self, batch, batch_idx):
        tensors, targets = batch
        outputs = self.model(tensors)
        loss = nn.functional.cross_entropy(outputs, targets)
        self.log("loss", loss, prog_bar=True, logger=True, on_step=True)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        tensors, targets = batch
        outputs = self.model(tensors)
        val_loss = nn.functional.cross_entropy(outputs, targets)
        self.log("val_loss", val_loss, prog_bar=True, logger=True, on_step=True)
        return {"loss", val_loss}

    def test_step(self, batch, batch_idx):
        tensors, targets = batch
        outputs = self.model(tensors)
        test_loss = nn.functional.cross_entropy(outputs, targets)
        self.log("test_loss", test_loss, prog_bar=True, logger=True, on_step=True)

    def predict_step(self, batch, batch_idx):
        tensors, targets = batch
        pred = self.model(tensors)
        return pred

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        return [optimizer], [lr_scheduler]
