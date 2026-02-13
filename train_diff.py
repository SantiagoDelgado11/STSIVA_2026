from utils.utils import save_images, save_metrics, AverageMeter
import wandb
from torchvision import transforms
import os
import torch
import torch.nn as nn
from tqdm import tqdm
from torch import optim
import logging
from utils.ddpm import Diffusion
from utils.utils import (
    save_checkpoint,
    load_checkpoint,
    cleanup_old_checkpoints,
)
from guided_diffusion.script_util import create_model
from torchvision import datasets
from torch.utils.data import DataLoader


def train(args):

    path_name = (
        f"e_{args.epochs}_bs_{args.batch_size}_lr_{args.lr}_seed_{args.seed}_img_{args.image_size}_schedule_{args.schedule_name}_gpu_{args.gpu_id}_c_{args.channels}_si_{args.save_img}"
    )
    if args.use_spectral_norm:
        path_name = f"{path_name}_sn_pi{args.spectral_norm_power_iters}"
    args.save_path = os.path.join(args.save_path, path_name)
    run_name = os.path.basename(args.save_path)
    images_path, model_path, metrics_path = save_metrics(args.save_path)

    # Create checkpoint directory
    checkpoint_dir = os.path.join(args.save_path, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    logging.basicConfig(
        filename=f"{metrics_path}/training.log",
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
    )

    logging.info(f"Starting training with parameters: {args}")

    wandb.login(key="b879bf20f3c31bfcf13289e363f4d3394f7d7671")
    wandb.init(project=args.project_name, name=run_name, config=args)

    device = args.device

    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
    ])

    train_dataset = datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )


    model = create_model(
        image_size=args.image_size,
        num_channels=64,
        num_res_blocks=3,
        input_channels=args.channels,
        use_spectral_norm=args.use_spectral_norm,
        spectral_norm_power_iters=args.spectral_norm_power_iters,
        spectral_norm_eps=args.spectral_norm_eps,
    ).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    mse = nn.MSELoss()
    diffusion = Diffusion(
        device=device,
        img_size=args.image_size,
        schedule_name=args.schedule_name,
        channels=args.channels,
    )

    # Initialize variables for checkpoint loading
    start_epoch = 0
    best_loss = float("inf")

    # Check if we're resuming from a checkpoint
    if args.resume:
        checkpoint_path = os.path.join(checkpoint_dir, args.resume)
        start_epoch, best_loss = load_checkpoint(checkpoint_path, model, optimizer, device)
        start_epoch += 1  # Start from the next epoch

    for epoch in range(start_epoch, args.epochs):
        train_loss = AverageMeter()
        data_loop_train = tqdm(enumerate(train_loader), total=len(train_loader), colour="red")

        for _, (images, _) in data_loop_train:
            images = train_data
            images = images.to(device)

            images = images * 2 - 1

            t = diffusion.sample_timesteps(images.shape[0]).to(device)
            x_t, noise = diffusion.noise_images(images, t)
            predicted_noise = model(x_t, t)
            loss = mse(noise, predicted_noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss.update(loss.item())
            data_loop_train.set_postfix(loss=train_loss.avg)

        logging.info(f"Epoch {epoch} loss: {train_loss.avg}")

        # Save checkpoint only at specified intervals or if it's the best model
        if (epoch + 1) % args.checkpoint_interval == 0 or epoch == args.epochs - 1:
            checkpoint_state = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "best_loss": best_loss,
                "rng_state": torch.get_rng_state(),
                "args": vars(args),
            }

            if device == "cuda":
                checkpoint_state["cuda_rng_state"] = torch.cuda.get_rng_state()

            checkpoint_filename = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pth.tar")
            save_checkpoint(checkpoint_state, checkpoint_filename)

            # Clean up old checkpoints (keep only the most recent ones)
            cleanup_old_checkpoints(checkpoint_dir, keep_last=args.keep_checkpoints)

        # Always save the latest checkpoint for easy resuming
        latest_checkpoint = os.path.join(checkpoint_dir, "latest.pth.tar")
        save_checkpoint(checkpoint_state, latest_checkpoint)

        if (epoch + 1) % args.save_img == 0:

            sampled_images = diffusion.sample(model, n=1)
            save_images(sampled_images, f"{images_path}/epoch_{epoch}_sampled.png")

        wandb.log(
            {
                "epoch": epoch,
                "sampled_images": (wandb.Image(sampled_images) if (epoch + 1) % args.save_img == 0 else None),
                "train_loss": train_loss.avg,
            }
        )


def launch():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--channels", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--save_path", type=str, default="weights/")
    parser.add_argument("--save_img", type=int, default=100, help="Save images every N epochs")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--project_name", type=str, default="CAMSAP")
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint file to resume training (e.g., 'latest.pth.tar')",
    )
    parser.add_argument(
        "--checkpoint_interval",
        type=int,
        default=1,
        help="Save checkpoint every N epochs (default: 5)",
    )
    parser.add_argument(
        "--keep_checkpoints",
        type=int,
        default=3,
        help="Number of recent checkpoints to keep (default: 3)",
    )
    parser.add_argument("--image_size", type=int, default=32)
    parser.add_argument(
        "--schedule_name",
        type=str,
        default="cosine",
        choices=["linear", "cosine"],
        help="Noise schedule name (default: 'linear')",
    )

    parser.add_argument("--gpu_id", type=str, default="0")
    parser.add_argument(
        "--use_spectral_norm",
        type=bool,
        default=True,
        help="Apply spectral normalization to Conv/Linear layers.",
    )
    parser.add_argument(
        "--spectral_norm_power_iters",
        type=int,
        default=1,
        help="Power iterations for spectral normalization.",
    )
    parser.add_argument(
        "--spectral_norm_eps",
        type=float,
        default=1e-12,
        help="Epsilon for spectral normalization.",
    )

    args = parser.parse_args()
    args.device = f"cuda:{args.gpu_id}"
    train(args)


if __name__ == "__main__":
    launch()
