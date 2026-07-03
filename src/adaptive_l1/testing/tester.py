import mrpro
import torch

from tqdm import tqdm
from adaptive_l1.testing.statistics import brain_mask, psnr

import csv


def test_model(model, test_loader, device, run_dir, metrics_fname):

    metrics_file = run_dir / f"{metrics_fname}.csv"
    model.eval()
    with torch.no_grad():

        mse_values_list = []
        ssim_values_list = []
        psnr_values_list = []

        test_n_samples = 0

        for batch in tqdm(
            test_loader,
            desc="test loop",
            position=0,
            leave=False,
            disable=False,
        ):
            kdata = batch["kdata"].to(device)
            adjoint = batch["adjoint"].to(device)
            mask = batch["mask"].to(device)
            target = batch["target"].to(device)

            mask_operator = mrpro.operators.CartesianMaskingOp(mask)

            recon = model(adjoint, kdata, mask_operator)

            batch_size = target.shape[0]
            target_mask = (
                torch.stack(
                    [
                        brain_mask(target[k].abs().squeeze(), threshold=0.1)
                        for k in range(batch_size)
                    ],
                    dim=0,
                )
                .unsqueeze(1)
                .unsqueeze(1)
            )

            mse_metric = mrpro.operators.functionals.MSE(
                target=target, weight=target_mask, keepdim=True
            )
            (mse_value,) = mse_metric(recon)

            ssim_metric = mrpro.operators.functionals.SSIM(
                target=target, weight=target_mask, reduction="volume"
            )

            (ssim_value,) = ssim_metric(recon)

            psnr_value = psnr(target, recon, weight=target_mask, reduction="full")

            psnr_values_list.extend(psnr_value.tolist())
            mse_values_list.extend(mse_value.tolist())
            ssim_values_list.extend(ssim_value.squeeze(1).tolist())

            test_n_samples += batch_size

        psnr_average = torch.mean(torch.tensor(psnr_values_list))
        psnr_std = torch.std(torch.tensor(psnr_values_list))

        mse_average = torch.mean(torch.tensor(mse_values_list))
        mse_std = torch.std(torch.tensor(mse_values_list))

        ssim_average = torch.mean(torch.tensor(ssim_values_list))
        ssim_std = torch.std(torch.tensor(ssim_values_list))

        with open(metrics_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([f"Tested on {test_n_samples} samples"])
            writer.writerow(
                [
                    f"psnr (mean +- std): {psnr_average} +- {psnr_std}",
                ]
            )
            writer.writerow(
                [
                    f"mse (mean +- std): {mse_average} +- {mse_std}",
                ]
            )
            writer.writerow(
                [
                    f"ssim (mean +- std): {ssim_average} +- {ssim_std}",
                ]
            )
