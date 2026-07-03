import mrpro
import torch
from tqdm import tqdm
import itertools


def forward_pass(model, batch, device):

    kdata = batch["kdata"].to(device)
    adjoint = batch["adjoint"].to(device)
    mask = batch["mask"].to(device)
    target = batch["target"].to(device)

    mask_operator = mrpro.operators.CartesianMaskingOp(mask)

    recon = model(adjoint, kdata, mask_operator)

    return recon, target


def train_model(
    model,
    training_loader,
    validation_loader,
    optimizer,
    loss_function,
    device,
    n_epochs,
    run_dir,
    config,
):
    use_wandb = False
    try:
        import wandb

        use_wandb = True
    except ImportError:
        print("wandb not installed, only the validation mse will be logged")

    if use_wandb:
        import matplotlib.pyplot as plt

        wandb.init(project="AdaptiveL1", config=config, name=config["experiment_name"])

    else:
        import csv

        metrics_file = run_dir / "training_metrics.csv"
        with open(metrics_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "epoch",
                    "validation_loss",
                ]
            )

    best_val_loss = float("inf")
    for epoch in tqdm(
        range(n_epochs),
        desc="epoch loop",
        position=0,
        disable=False,
    ):
        model.train()
        for batch in tqdm(
            training_loader,
            desc=" training batch loop",
            position=1,
            leave=False,
            disable=False,
        ):
            recon, target = forward_pass(model, batch, device)
            loss = loss_function(
                torch.view_as_real(recon),
                torch.view_as_real(target),
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        model.eval()
        validation_loss_sum = 0.0
        validation_n_samples = 0

        with torch.no_grad():
            for batch in tqdm(
                validation_loader,
                desc="evaluation loop",
                position=1,
                leave=False,
                disable=False,
            ):
                recon, target = forward_pass(model, batch, device)
                loss = loss_function(
                    torch.view_as_real(recon),
                    torch.view_as_real(target),
                )

                batch_size = target.shape[0]
                validation_loss_sum += loss.item() * batch_size
                validation_n_samples += batch_size

            validation_loss = validation_loss_sum / validation_n_samples

        if use_wandb:
            sample_id = 5  # just picked for demonstration purposes
            batch = next(itertools.islice(validation_loader, sample_id, sample_id + 1))

            with torch.no_grad():
                recon, target = forward_pass(model, batch, device)

            wandb.log({"validation-loss": validation_loss}, step=epoch + 1)
            fig, ax = plt.subplots(1, 3, figsize=(1 * 15, 3 * 15))
            recons_list = [
                batch["adjoint"].detach().cpu(),
                recon.detach().cpu(),
                batch["target"].detach().cpu(),
            ]
            titles_list = ["adjoint", "recon", "target"]
            for k, (recon_, title) in enumerate(zip(recons_list, titles_list)):
                ax[k].imshow(
                    recon_.abs()[0].squeeze(),
                    cmap="gray",
                    clim=[0, 0.4 * target.abs().max()],
                )
                mse = torch.nn.functional.mse_loss(
                    torch.view_as_real(recon_), torch.view_as_real(recons_list[-1])
                ).item()
                ax[k].set_title(title)
                if k < 2:
                    ax[k].text(
                        0.5,
                        0.12,
                        f"MSE: {mse:.2e}",
                        color="yellow",
                        fontsize=24,
                        horizontalalignment="center",
                        verticalalignment="top",
                        transform=ax[k].transAxes,
                        bbox={
                            "facecolor": "black",
                            "alpha": 0.8,
                            "pad": 1,
                        },
                    )

            plt.setp(ax, xticks=[], yticks=[])
            wandb.log({"training-figure": fig})
            plt.close()
        else:
            with open(metrics_file, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        epoch + 1,
                        validation_loss,
                    ]
                )

        latest_checkpoint = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "validation_loss": validation_loss,
            "config": config,
        }

        if validation_loss < best_val_loss:
            best_val_loss = validation_loss

            torch.save(
                latest_checkpoint,
                run_dir / "model.pt",
            )

            if use_wandb:
                wandb.run.summary["best_val_loss"] = best_val_loss
                wandb.run.summary["best_epoch"] = epoch + 1

        training_loader.dataset.set_epoch(epoch + 1)

    if use_wandb:
        wandb.finish()
