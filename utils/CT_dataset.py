import torch
import time
from torchvision import transforms
from torch.utils.data import SubsetRandomSampler
import os
import torch
import numpy as np
import random
import os
import h5py
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class CTDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.file_paths = [
            os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".hdf5")
        ]
        self.transform = transform
        self.image_indices = []

        for file_index, file_path in enumerate(self.file_paths):
            with h5py.File(file_path, "r") as h5_file:
                num_images = h5_file["data"].shape[0]
                self.image_indices.extend([(file_index, i) for i in range(num_images)])

    def __len__(self):
        return len(self.image_indices)

    def __getitem__(self, idx):
        file_index, image_index = self.image_indices[idx]
        file_path = self.file_paths[file_index]

        with h5py.File(file_path, "r") as h5_file:
            image = h5_file["data"][image_index]

        if self.transform:
            image = self.transform(image)

        return image


class LoDoPaB:
    def __init__(self, batch_size, workers, im_size=None):
        self.batch_size = batch_size
        self.workers = workers
        if im_size is None:
            self.transform = transforms.Compose([transforms.ToTensor()])
        else:
            self.transform = transforms.Compose([transforms.ToTensor(), transforms.Resize(im_size)])

    def get_loaders(self):
        train_dataset = CTDataset("data/ground_truth_train/", transform=self.transform)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.workers,
            pin_memory=True,
        )

        test_dataset = CTDataset("data/ground_truth_test/", transform=self.transform)
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.workers,
            pin_memory=True,
        )

        val_dataset = CTDataset("data/ground_truth_validation/", transform=self.transform)
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.workers,
            pin_memory=True,
        )

        return train_loader, val_loader, test_loader


def set_seed(seed: int):
    # os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # torch.use_deterministic_algorithms(mode=True)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_time():

    return str(time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime()))


def save_metrics(save_path):

    images_path = save_path + "/images"
    model_path = save_path + "/model"
    metrics_path = save_path + "/metrics"

    os.makedirs(save_path, exist_ok=True)
    os.makedirs(images_path, exist_ok=True)
    os.makedirs(model_path, exist_ok=True)
    os.makedirs(metrics_path, exist_ok=True)

    return images_path, model_path, metrics_path


def save_npy_metric(file, metric_name):

    with open(f"{metric_name}.npy", "wb") as f:
        np.save(f, file)


def count_parameters(model):

    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":

    import matplotlib.pyplot as plt
    from torchvision.utils import make_grid

    train_loader, val_loader, test_loader = LoDoPaB(
        batch_size=32,
        workers=4,
    ).get_loaders()

    batch = next(iter(train_loader))
    grid = make_grid(batch, nrow=8, normalize=True)

    plt.imshow(grid.permute(1, 2, 0))
    plt.axis("off")

    plt.show()

    print(len(train_loader), len(val_loader), len(test_loader))
    print(len(train_loader.dataset), len(val_loader.dataset), len(test_loader.dataset))
    print(train_loader.dataset[0].shape)
    print(train_loader.dataset[0].max(), train_loader.dataset[0].min())
