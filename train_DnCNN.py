import torch
import torch.nn as nn
import argparse
from utils.utils import print_dict, set_seed, save_metrics, AverageMeter, save_npy_metric
import os
from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio
import logging
from utils.CT_dataset import LoDoPaB
import numpy as np
from tqdm import tqdm
import wandb
from utils.DnCNN import DnCNN

def main(args):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    set_seed(args.seed)

    path_name = f"CT_SPECTRAL_NORM_lr_{args.lr}_b_{args.batch_size}_e_{args.epochs}_sd_{args.seed}_im_{args.im_size}_ml_{args.milestone}"

    args.save_path = args.save_path + path_name
    if os.path.exists(f"{args.save_path}/metrics/metrics.npy"):
        print("Experiment already done")
        exit()

    images_path, model_path, metrics_path = save_metrics(f"{args.save_path}")
    current_psnr = 0

    logging.basicConfig(
        filename=f"{metrics_path}/training.log",
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
    )

    logging.info(f"Starting training with parameters: {args}")

    loss_train_record = np.zeros(args.epochs)
    ssim_train_record = np.zeros(args.epochs)
    psnr_train_record = np.zeros(args.epochs)
    loss_val_record = np.zeros(args.epochs)
    ssim_val_record = np.zeros(args.epochs)
    psnr_val_record = np.zeros(args.epochs)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    SSIM = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    PSNR = PeakSignalNoiseRatio(data_range=1.0).to(device)

    im_size = (args.im_size, args.im_size)

    dataset_CT = LoDoPaB(batch_size=args.batch_size, workers=0, im_size=im_size)

    train_loader, val_loader, test_loader = dataset_CT.get_loaders()

    dcnn = DnCNN(channels=1, num_of_layers=20).to(device)
    optimizer = torch.optim.Adam(dcnn.parameters(), lr=args.lr)

    criterion = nn.MSELoss(size_average=False).to(device)
    noiseL_B = [0, 55]

    wandb.login(key="b879bf20f3c31bfcf13289e363f4d3394f7d7671")
    wandb.init(project=args.project_name, name=path_name, config=args)

    for epoch in range(args.epochs):

        if epoch < args.milestone:
            current_lr = args.lr
        else:
            current_lr = args.lr / 10.0

        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr
        print("learning rate %f" % current_lr)

        dcnn.train()

        train_loss = AverageMeter()
        train_ssim = AverageMeter()
        train_psnr = AverageMeter()

        data_loop_train = tqdm(enumerate(train_loader), total=len(train_loader), colour="red")
        for _, train_data in data_loop_train:

            img = train_data
            img = img.to(device)

            noise = torch.zeros(img.size())
            stdN = np.random.uniform(noiseL_B[0], noiseL_B[1], size=noise.size()[0])
            for n in range(noise.size()[0]):
                sizeN = noise[0, :, :, :].size()
                noise[n, :, :, :] = torch.FloatTensor(sizeN).normal_(mean=0, std=stdN[n] / 255.0)

            noise = noise.to(device)
            noisy_img = img + noise

            predicted_noise = dcnn(noisy_img)

            loss_train = criterion(predicted_noise, noise) / (img.size()[0] * 2)

            optimizer.zero_grad()
            loss_train.backward()
            optimizer.step()

            train_loss.update(loss_train.item())
            train_ssim.update(SSIM(noisy_img - predicted_noise, img).item())
            train_psnr.update(PSNR(noisy_img - predicted_noise, img).item())

            data_loop_train.set_description(f"Epoch {epoch + 1}/{args.epochs}")
            data_loop_train.set_postfix(
                loss=train_loss.avg, psnr=train_psnr.avg, ssim=train_ssim.avg
            )

        logging.info(
            f"Epoch {epoch + 1}/{args.epochs} - Train Loss: {train_loss.avg:.4f} - Train PSNR: {train_psnr.avg:.4f} - Train SSIM: {train_ssim.avg:.4f}"
        )

        val_loss = AverageMeter()
        val_ssim = AverageMeter()
        val_psnr = AverageMeter()
        data_loop_val = tqdm(enumerate(val_loader), total=len(val_loader), colour="green")
        with torch.no_grad():

            dcnn.eval()

            for _, val_data in data_loop_val:
                img = val_data
                img = img.to(device)

                noise = torch.zeros(img.size())
                stdN = np.random.uniform(noiseL_B[0], noiseL_B[1], size=noise.size()[0])
                for n in range(noise.size()[0]):
                    sizeN = noise[0, :, :, :].size()
                    noise[n, :, :, :] = torch.FloatTensor(sizeN).normal_(
                        mean=0, std=stdN[n] / 255.0
                    )

                noise = noise.to(device)
                noisy_img = img + noise

                predicted_noise = dcnn(noisy_img)

                loss_val = criterion(predicted_noise, noise) / (img.size()[0] * 2)

                val_loss.update(loss_val.item())
                val_ssim.update(SSIM(noisy_img - predicted_noise, img).item())
                val_psnr.update(PSNR(noisy_img - predicted_noise, img).item())

                data_loop_val.set_description(f"Epoch {epoch + 1}/{args.epochs}")
                data_loop_val.set_postfix(loss=val_loss.avg, psnr=val_psnr.avg, ssim=val_ssim.avg)

        logging.info(
            f"Epoch {epoch + 1}/{args.epochs} - Val Loss: {val_loss.avg:.4f} - Val PSNR: {val_psnr.avg:.4f} - Val SSIM: {val_ssim.avg:.4f}"
        )

        if val_psnr.avg > current_psnr:
            current_psnr = val_psnr.avg
            print(f"Saving model with PSNR: {current_psnr:.4f}")
            torch.save(dcnn.state_dict(), f"{model_path}/dcnn.pth")

        wandb.log(
            {
                "train_loss": train_loss.avg,
                "train_psnr": train_psnr.avg,
                "train_ssim": train_ssim.avg,
                "val_loss": val_loss.avg,
                "val_psnr": val_psnr.avg,
                "val_ssim": val_ssim.avg,
            }
        )

        loss_train_record[epoch] = train_loss.avg
        psnr_train_record[epoch] = train_psnr.avg
        ssim_train_record[epoch] = train_ssim.avg
        loss_val_record[epoch] = val_loss.avg
        psnr_val_record[epoch] = val_psnr.avg
        ssim_val_record[epoch] = val_ssim.avg

    test_loss = AverageMeter()
    test_ssim = AverageMeter()
    test_psnr = AverageMeter()

    del dcnn

    dcnn = DnCNN(channels=1, num_of_layers=20).to(device)

    dcnn.load_state_dict(torch.load(f"{model_path}/dcnn.pth", map_location=device))
    dcnn.eval()

    data_loop_test = tqdm(enumerate(test_loader), total=len(test_loader), colour="magenta")
    with torch.no_grad():

        for _, test_data in data_loop_test:
            img = test_data
            img = img.to(device)

            noise = torch.zeros(img.size())
            stdN = np.random.uniform(noiseL_B[0], noiseL_B[1], size=noise.size()[0])
            for n in range(noise.size()[0]):
                sizeN = noise[0, :, :, :].size()
                noise[n, :, :, :] = torch.FloatTensor(sizeN).normal_(mean=0, std=stdN[n] / 255.0)

            noise = noise.to(device)
            noisy_img = img + noise

            predicted_noise = dcnn(noisy_img)

            loss_test = criterion(predicted_noise, noise) / (img.size()[0] * 2)

            test_loss.update(loss_test.item())
            test_ssim.update(SSIM(noisy_img - predicted_noise, img).item())
            test_psnr.update(PSNR(noisy_img - predicted_noise, img).item())

            data_loop_test.set_description("TEST")
            data_loop_test.set_postfix(loss=test_loss.avg, psnr=test_psnr.avg, ssim=test_ssim.avg)

    logging.info(
        f"TEST - Loss: {test_loss.avg:.4f} - PSNR: {test_psnr.avg:.4f} - SSIM: {test_ssim.avg:.4f}"
    )

    save_npy_metric(
        dict(
            loss_train_record=loss_train_record,
            psnr_train_record=psnr_train_record,
            ssim_train_record=ssim_train_record,
            loss_val_record=loss_val_record,
            psnr_val_record=psnr_val_record,
            ssim_val_record=ssim_val_record,
            loss_test_record=test_loss.avg,
            psnr_test_record=test_psnr.avg,
            ssim_test_record=test_ssim.avg,
        ),
        f"{metrics_path}/metrics",
    )

    wandb.log(
        {
            "test_loss": test_loss.avg,
            "test_psnr": test_psnr.avg,
            "test_ssim": test_ssim.avg,
        }
    )

    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parameter Processing")
    parser.add_argument("--im_size", type=int, default=256, help="Image size")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=2**4)
    parser.add_argument("--save_path", type=str, default="WEIGHTS/DNCNN/")
    parser.add_argument("--project_name", type=str, default="CAMSAP")
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument(
        "--milestone",
        type=int,
        default=30,
        help="When to decay learning rate; should be less than epochs",
    )
    parser.add_argument("--gpu_id", type=str, default="0")
    args = parser.parse_args()
    args_dict = vars(args)

    print_dict(args_dict)
    main(args)
