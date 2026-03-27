import wandb
import argparse
import os

import matplotlib.pyplot as plt
import torch

from algos.ddnm import DDNM
from algos.diffpir import DiffPIR
from algos.dps import DPS
from guided_diffusion.script_util import create_model
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchvision import transforms
from utils.SPC_model import SPCModel
from torchvision.datasets import CIFAR10
from utils.utils import set_seed


def main(opt):

    print("Options:")
    for key, value in vars(opt).items():
        print(f"{key}: {value}")

    device = f"cuda:{opt.gpu_id}" if torch.cuda.is_available() else "cpu"
    set_seed(7)

    # ############################## MODEL ############################

    ckpt = torch.load(opt.weights, map_location=device, weights_only=True)
    net = create_model(image_size=opt.image_size, num_channels=64, num_res_blocks=3, input_channels=3).to("cuda")
    net.load_state_dict(ckpt["model_state"])
    net.eval()

    # ############################## DATASET ############################

    dataset = CIFAR10(
        root="data",
        train=False,  # Usamos el set de prueba
        download=True,  # Descarga el dataset automáticamente si no lo encuentra
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        ),
    )

    # ############################## IMAGE ############################

    GT = dataset[opt.idx][0].unsqueeze(0).to(device)

    # Normalize the image to [-1, 1] range

    GT = GT * 2 - 1

    # ####################### FORWARD AND TRANSPOSE PASS #######################

    inverse_model = SPCModel(im_size=opt.image_size, compression_ratio=opt.sampling_ratio)

    y = inverse_model.forward_pass(GT)
    x_estimate = inverse_model.transpose_pass(y)

    if opt.algo == "DPS":

        diff = DPS(
            device=device,
            img_size=opt.image_size,
            noise_steps=1000,
            schedule_name="cosine",
            channels=3,
            scale=opt.dps_scale,
            clip_denoised=False,
        )

        reconstruction = diff.sample(
            model=net,
            y=y,
            forward_pass=inverse_model.forward_pass,
        )

    elif opt.algo == "DDNM":
        diff = DDNM(
            device=device,
            img_size=opt.image_size,
            noise_steps=1000,
            schedule_name="cosine",
            channels=3,
            eta=opt.ddnm_eta,
        )

        reconstruction = diff.sample(
            model=net,
            y=y,
            forward_pass=inverse_model.forward_pass,
            pseudo_inverse=inverse_model.pseudo_inverse,
            ground_truth=GT,
            track_metrics=opt.plot_metrics == "True",
        )

    elif opt.algo == "DiffPIR":
        diff = DiffPIR(
            device=device,
            img_size=opt.image_size,
            noise_steps=1000,
            schedule_name="cosine",
            channels=3,
            cg_iters=opt.CG_iters_diffpir,
            noise_level_img=opt.noise_level_img,
            iter_num=opt.iter_num,
            eta=0,
            zeta=1,
        )

        reconstruction = diff.sample(
            model=net,
            y=y,
            forward_pass=inverse_model.forward_pass,
            transpose_pass=inverse_model.transpose_pass,
        )

    else:
        raise ValueError("Invalid algorithm specified. Choose from 'DPS', 'DDNM', or 'DiffPIR'.")

    SSIM = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    PSNR = PeakSignalNoiseRatio(data_range=1.0).to(device)

    # Normalize the images to [0, 1] range for SSIM and PSNR calculations
    reconstruction = (reconstruction + 1) / 2  # Predicted image from the diffusion model
    GT = (GT + 1) / 2  # Ground truth image

    reconstruction = reconstruction.clamp(0, 1)
    GT = GT.clamp(0, 1)

    # ########### ERROR MAP ####################

    error_map = (reconstruction - GT).abs().mean(dim=1, keepdim=True)

    # Calculate SSIM and PSNR for the predicted and estimated images

    ssim_pred = SSIM(reconstruction, GT)
    psnr_pred = PSNR(reconstruction, GT)
    consistency = torch.linalg.norm(inverse_model.forward_pass(reconstruction) - inverse_model.forward_pass(GT)).item()
    error = torch.linalg.norm(reconstruction - GT).item()

    # Plotting the results

    fig, ax = plt.subplots(1, 4, figsize=(25, 5))
    ax[0].imshow(GT[0].permute(1, 2, 0).cpu().detach().numpy())
    ax[0].axis("off")
    ax[0].set_title("Ground Truth")

    ax[1].imshow(x_estimate[0].permute(1, 2, 0).cpu().detach().numpy())
    ax[1].axis("off")
    ax[1].set_title("A^T y")

    ax[2].imshow(reconstruction[0].permute(1, 2, 0).cpu().detach().numpy())
    ax[2].axis("off")
    ax[2].set_title(f"{opt.algo} Predicted\nSSIM: {ssim_pred:.4f}, PSNR: {psnr_pred:.2f}")


    ax[3].imshow(error_map[0, 0].cpu().detach().numpy(), cmap="hot")
    ax[3].axis("off")
    ax[3].set_title(f"Error Map ({opt.algo} - GT)")
    plt.colorbar(ax[3].images[0], ax=ax[3])
    plt.suptitle(f"SPC Reconstruction with {opt.algo} Algorithm", fontsize=16)
    plt.tight_layout()

    if opt.save_image == "True":
        save_dir = f"results/SPC/{opt.algo}/{opt.sampling_method}/{opt.sampling_ratio:.2f}/{opt.idx}"
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(
            f"{save_dir}/reconstruction_idx_{opt.idx}_{psnr_pred:.2f}_{ssim_pred:.3f}.pdf",
            bbox_inches="tight",
        )

    if opt.plot == "True":
        plt.show()
    else:
        plt.close()

    if opt.use_wandb == "True":
        wandb.login(key=opt.wandb_id)
        wandb.init(
            project=opt.project_name,
            name=opt.name,
            config=vars(opt),
        )
        wandb.log(
            {
                "SSIM": ssim_pred.item(),
                "PSNR": psnr_pred.item(),
                "Consistency": consistency,
                "Error": error,
            }
        )

    print(f"SSIM ({opt.algo}): {ssim_pred:.3f}, PSNR ({opt.algo}): {psnr_pred:.2f}, Consistency: {consistency:.2f}, Error: {error:.2f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--idx", type=int, default=10)
    p.add_argument(
        "--weights",
        type=str,
        default="weights/e_1000_bs_64_lr_0.0003_seed_2_img_32_schedule_cosine_gpu_0_c_3_si_100/checkpoints/latest.pth.tar",
    )
    p.add_argument("--batch_size", type=int, default=150, help="Batch size for sampling (default: 1)")
    p.add_argument("--image_size", type=int, default=32)
    p.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--gpu_id", type=int, default=0, help="GPU ID to use for training (default: 0)")
    p.add_argument("--save_image", type=str, default="False", choices=["True", "False"])
    p.add_argument("--plot", type=str, default="True", choices=["True", "False"])
    p.add_argument(
        "--plot_metrics",
        type=str,
        default="False",
        choices=["True", "False"],
        help="Plot PSNR, SSIM, consistency, disagreement and fixed point error during sampling",
    )

    ################# SAMPLING / ALGO PARAMS #################

    p.add_argument("--sampling_ratio", type=float, default=1 / 2)

    p.add_argument(
        "--algo",
        type=str,
        default="DiffPIR",
        choices=["DPS", "DDNM", "DiffPIR"],
    )

    ####################### DDNM PARAMS ########################
    p.add_argument("--ddnm_eta", type=float, default=1, help="DDNM stochasticity parameter")

    ####################### DPS PARAMS  ########################

    p.add_argument("--dps_scale", type=float, default=0.0125, help="DPS scale factor")

    p.add_argument(
        "--skip_type",
        type=str,
        default="uniform",
        choices=["uniform", "quad"],
        help="Schedule type for step skipping",
    )
    p.add_argument("--iter_num", type=int, default=1000, help="Number of steps when using quad schedule")

    ###################### DIFPIR PARAMS #########################

    p.add_argument("--CG_iters_diffpir", type=int, default=5)
    p.add_argument("--noise_level_img", type=float, default=0.0)

    ###################### WANDB PARAMS #########################
    p.add_argument(
        "--use_wandb",
        type=str,
        default="False",
        choices=["True", "False"],
        help="Use Weights & Biases for logging",
    )
    p.add_argument("--wandb_id", type=str, default="b879bf20f3c31bfcf13289e363f4d3394f7d7671")
    p.add_argument("--project_name", type=str, default="STSIVA_2026")
    p.add_argument("--name", type=str, default="run_one_img")

    main(p.parse_args())