import random
import sys

import mlflow
import numpy as np
import torch
from mlflow.models.signature import infer_signature
from torch import nn, optim
from torchinfo import summary
from torchmetrics import Accuracy

from model import init_model
from setup_data import create_dataloader

if __name__ == "__main__":
    # Get params
    experiment_id = 0
    model_name = "pytorch-template"
    tracking_uri = "http://127.0.0.1:8080"
    disable_cuda = False
    cuda_device = 0
    device = torch.device(
        "cpu"
        if disable_cuda or not torch.cuda.is_available()
        else f"cuda:{cuda_device}"
    )
    seed = 0
    epochs = 10
    learning_rate = 0.005
    batch_size = 32
    log_interval = 100

    mlflow.set_tracking_uri(tracking_uri)
    with mlflow.start_run(experiment_id=experiment_id) as run:
        # Set seed to improve reproduciblity
        random.seed(seed)
        torch.random.manual_seed(seed)
        if not disable_cuda:
            torch.cuda.manual_seed(seed)

        # Prepare Train/Test loaders
        train_loader, val_loader = create_dataloader(
            batch_size=batch_size,
            **({"num_workers": 1, "pin_memory": True} if not disable_cuda else {}),
        )

        # Initialize model
        model = init_model(device=device)

        # Log model summary.
        with open("model_summary.txt", "w") as f:
            f.write(str(summary(model)))
        mlflow.log_artifact("model_summary.txt")

        # Initialize optimizer, criterion, scheduler, metric_fn
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
        criterion = nn.NLLLoss().to(device)
        metric_fn = Accuracy(task="multiclass", num_classes=10).to(device)

        params = {
            "seed": seed,
            "device": device,
            "epochs": epochs,
            "lr": learning_rate,
            "batch_size": batch_size,
            "optimizer": optimizer.__class__.__name__,
            "scheduler": scheduler.__class__.__name__,
            "loss_fn": criterion.__class__.__name__,
            "metric_fn": metric_fn.__class__.__name__,
            "log_interval": log_interval,
        }
        mlflow.log_params(params)

        for epoch in range(0, epochs):
            print(f"Epoch {epoch+1}\n-------------------------------")

            # Train
            num_batches = len(train_loader)
            model.train()
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)

                output = model(data)
                loss = criterion(output, target)
                accuracy = metric_fn(output, target)

                # Backprop
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                if batch_idx % params["log_interval"] == 0:
                    loss, current = loss.item(), batch_idx
                    step = batch_idx // 100 * (epoch + 1)
                    mlflow.log_metric("loss", f"{loss:2f}", step=step)
                    mlflow.log_metric("accuracy", f"{accuracy:2f}", step=step)
                    print(
                        f"loss: {loss:2f} accuracy: {accuracy:2f} [{current} / {len(train_loader)}]"
                    )

            # Validation
            num_batches = len(val_loader)
            model.eval()
            val_loss, val_accuracy = 0, 0
            with torch.no_grad():
                for batch_idx, (data, target) in enumerate(val_loader):
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    loss = criterion(output, target)
                    val_loss += loss.item()
                    val_accuracy += metric_fn(output, target)

            val_loss /= num_batches
            val_accuracy /= num_batches
            mlflow.log_metric("val_loss", f"{val_loss:2f}", step=0)
            mlflow.log_metric("val_accuracy", f"{val_accuracy:2f}", step=0)
            print(
                f"\nValidation metrics: \nAccuracy: {val_accuracy:.2f}, Avg loss: {val_loss:2f} \n"
            )

        # Infer the signature of the model
        sample_input = next(iter(val_loader))[0]
        model.eval()
        with torch.no_grad():
            sample_output = model(sample_input.to(device))
        signature = infer_signature(sample_input.numpy(), sample_output.cpu().numpy())
        model_info = mlflow.pytorch.log_model(model, model_name, signature=signature)
