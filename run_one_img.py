import wandb
import argparse
import os

import matplotlib.pyplot as plt
import torch

from algos.ddnm import DDNM
from algos.DICE import DICE
from algos.diffpir import DiffPIR
from algos.dps import DPS
from algos.Method import Method
from guided_diffusion.script_util import create_model
from torch.utils.data import DataLoader
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchvision import transforms
from utils.ct_model import CTModel
from utils.test_set_loader import TestDataset
from utils.utils import set_seed


def main(opt):

    print("Options:")
    for key, value in vars(opt).items():
        print(f"{key}: {value}")

    device = f"cuda:{opt.gpu_id}" if torch.cuda.is_available() else "cpu"
    set_seed(7)

    # ############################## MODEL ############################

    ckpt = torch.load(opt.weights, map_location=device, weights_only=True)
    net = create_model(image_size=opt.image_size, num_channels=64, num_res_blocks=3, input_channels=1).to("cuda")
    net.load_state_dict(ckpt["model_state"])
    net.eval()

    # ############################## DATASET ############################

    dataset = TestDataset(
        "data/test_imgs",
        transform=transforms.Compose(
            [
                transforms.ToTensor(),  # Convert numpy array to Tensor
                transforms.Resize((opt.image_size, opt.image_size)),
            ]
        ),
    )
    testloader = DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    # ############################## IMAGE ############################

    GT = next(iter(testloader))[opt.idx].unsqueeze(0).to(device)

    # Normalize the image to [-1, 1] range

    GT = GT * 2 - 1

    # ####################### FORWARD AND TRANSPOSE PASS #######################

    inverse_model = CTModel(
        im_size=opt.image_size,
        num_angles=opt.num_angles,
        sampling_ratio=opt.sampling_ratio,
        sampling_method=opt.sampling_method,
    ).to(device)
    y = inverse_model.forward_pass(GT)
    x_estimate = inverse_model.transpose_pass(y)

    if opt.algo == "DPS":

        diff = DPS(
            device=device,
            img_size=opt.image_size,
            noise_steps=1000,
            schedule_name="cosine",
            channels=1,
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
            channels=1,
            eta=opt.ddnm_eta,
        )

        reconstruction = diff.sample(
            model=net,
            y=y,
            forward_pass=inverse_model.forward_pass,
            pseudo_inverse=inverse_model.pseudoinverse_cgls,
            ground_truth=GT,
            track_metrics=opt.plot_metrics == "True",
        )

    elif opt.algo == "Method":
        diff = Method(
            device=device,
            img_size=opt.image_size,
            noise_steps=1000,
            schedule_name="cosine",
            channels=1,
        )

        reconstruction = diff.sample(
            model=net,
            y=y,
            forward_pass=inverse_model.forward_pass,
            transpose_pass=inverse_model.transpose_pass,
            CG_iter=opt.CG_iters_diffpir,
        )

    elif opt.algo == "DiffPIR":
        diff = DiffPIR(
            device=device,
            img_size=opt.image_size,
            noise_steps=1000,
            schedule_name="linear",
            channels=1,
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

    elif opt.algo == "PnP_FISTA":
        from utils.DnCNN import DnCNN
        from algos.pnp_fista import PnPFISTA

        dncnn = DnCNN(channels=1, num_of_layers=20).to(device)
        dncnn.load_state_dict(torch.load(opt.dncnn_weights, map_location=device))
        dncnn.eval()

        diff = PnPFISTA(
            denoiser=dncnn,
            max_iter=opt.fista_iter,
            step_size=opt.fista_step,
            denoiser_strength=opt.denoiser_strength,
            device=device,
        )

        reconstruction = diff.sample(
            y=y,
            forward_pass=inverse_model.forward_pass,
            transpose_pass=inverse_model.transpose_pass,
        )

    elif opt.algo == "DICE":
        diff = DICE(
            device=device,
            img_size=opt.image_size,
            noise_steps=1000,
            schedule_name="cosine",
            channels=1,
            rho=opt.dice_rho,
            mu=opt.dice_mu,
            skip_type=opt.skip_type,
            iter_num=opt.iter_num,
        )

        reconstruction = diff.sample(
            model=net,
            y=y,
            forward_pass=inverse_model.forward_pass,
            transpose_pass=inverse_model.transpose_pass,
            ground_truth=GT,
            track_metrics=opt.plot_metrics == "True",
        )

    elif opt.algo == "FBP":
        reconstruction = inverse_model.pseudoinverse_cgls(y)
    else:
        raise ValueError("Invalid algorithm specified. Choose from 'DPS', 'DDNM', 'DiffPIR', 'PnP_FISTA', 'FBP', 'DICE', or 'Method'.")

    # ############################# METRICS ############################

    SSIM = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    PSNR = PeakSignalNoiseRatio(data_range=1.0).to(device)

    # Normalize the images to [0, 1] range for SSIM and PSNR calculations
    reconstruction = (reconstruction + 1) / 2  # Predicted image from the diffusion model
    GT = (GT + 1) / 2  # Ground truth image

    reconstruction = reconstruction.clamp(0, 1)
    GT = GT.clamp(0, 1)

    # ########### ERROR MAP ####################

    error_map = (reconstruction - GT).abs()

    # Calculate SSIM and PSNR for the predicted and estimated images

    ssim_pred = SSIM(reconstruction, GT)
    psnr_pred = PSNR(reconstruction, GT)
    consistency = torch.linalg.norm(inverse_model.forward_pass(reconstruction) - inverse_model.forward_pass(GT)).item()
    error = torch.linalg.norm(reconstruction - GT).item()

    # Plotting the results

    fig, ax = plt.subplots(1, 5, figsize=(25, 5))
    ax[0].imshow(GT[0, 0].cpu().detach().numpy(), cmap="gray")
    ax[0].axis("off")
    ax[0].set_title("Ground Truth")

    ax[1].imshow(x_estimate[0, 0].cpu().detach().numpy(), cmap="gray")
    ax[1].axis("off")
    ax[1].set_title("A^T y")

    ax[2].imshow(reconstruction[0, 0].cpu().detach().numpy(), cmap="gray")
    ax[2].axis("off")
    ax[2].set_title(f"{opt.algo} Predicted\nSSIM: {ssim_pred:.4f}, PSNR: {psnr_pred:.2f}")

    meas_vis = y[0, 0].cpu().detach().numpy().T
    meas_cmap = "jet"
    meas_title = f"Sinogram + {opt.sampling_method} Sampling | {opt.sampling_ratio:.2f} ratio"

    ax[3].imshow(meas_vis, cmap=meas_cmap)
    ax[3].axis("off")
    ax[3].set_title(meas_title)

    ax[4].imshow(error_map[0, 0].cpu().detach().numpy(), cmap="hot")
    ax[4].axis("off")
    ax[4].set_title(f"Error Map ({opt.algo} - GT)")
    plt.colorbar(ax[4].images[0], ax=ax[4])
    plt.suptitle(f"CT Reconstruction with {opt.algo} Algorithm", fontsize=16)
    plt.tight_layout()

    if opt.save_image == "True":
        save_dir = f"results/CT/{opt.algo}/{opt.sampling_method}/{int(opt.sampling_ratio * 180)}/{opt.idx}"
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(
            f"{save_dir}/reconstruction_idx_{opt.idx}_{psnr_pred:.2f}_{ssim_pred:.3f}.pdf",
            bbox_inches="tight",
        )

    if opt.plot == "True":
        plt.show()
    else:
        plt.close()

    plt.imshow(meas_vis, cmap=meas_cmap)
    plt.axis("off")
    meas_title_full = f"Sinogram + {opt.sampling_method} Sampling | {opt.sampling_ratio:.2f} ratio | algo: {opt.algo}"
    plt.title(meas_title_full)

    if opt.save_image == "True":
        plt.savefig(
            f"{save_dir}/measurement_idx_{opt.idx}_{psnr_pred:.2f}_{ssim_pred:.3f}.pdf",
            bbox_inches="tight",
        )

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
        default="weights/d_lodo_e_1000_bs_4_lr_0.0003_seed_2_img_256_schedule_cosine_gpu_1_c_1_si_100/checkpoints/latest.pth.tar",
    )
    p.add_argument("--batch_size", type=int, default=150, help="Batch size for sampling (default: 1)")
    p.add_argument("--image_size", type=int, default=256)
    p.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    p.add_argument("--num_angles", type=int, default=180)
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

    p.add_argument("--sampling_ratio", type=float, default=1 / 6)
    p.add_argument(
        "--sampling_method",
        type=str,
        default="uniform",
        choices=["uniform", "non_uniform", "limited"],
    )

    p.add_argument(
        "--algo",
        type=str,
        default="Method",
        choices=["DPS", "DDNM", "DiffPIR", "PnP_FISTA", "FBP", "DICE", "Method"],
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

    p.add_argument("--CG_iters_diffpir", type=int, default=20)
    p.add_argument("--noise_level_img", type=float, default=0.0)

    ###################### DICE PARAMS #########################

    p.add_argument("--dice_rho", type=float, default=0.9, help="DICE rho parameter")
    p.add_argument("--dice_mu", type=float, default=0.1, help="DICE mu (relaxation) parameter")

    ###################### PnP-FISTA PARAMS #########################
    p.add_argument(
        "--dncnn_weights",
        type=str,
        default="weights/DNCNN/CT_SPECTRAL_NORM_lr_0.001_b_16_e_150_sd_2_im_256_ml_30/model/dcnn.pth",
    )
    p.add_argument("--fista_iter", type=int, default=600)
    p.add_argument("--fista_step", type=float, default=0.001)
    p.add_argument(
        "--denoiser_strength",
        type=float,
        default=0.00001,
        help="Scale factor for the denoiser output in PnP-FISTA",
    )

    ###################### WANDB PARAMS #########################
    p.add_argument(
        "--use_wandb",
        type=str,
        default="False",
        choices=["True", "False"],
        help="Use Weights & Biases for logging",
    )
    p.add_argument("--wandb_id", type=str, default="b879bf20f3c31bfcf13289e363f4d3394f7d7671")
    p.add_argument("--project_name", type=str, default="CAMSAP_RUNS")
    p.add_argument("--name", type=str, default="run_one_img")

    main(p.parse_args())
