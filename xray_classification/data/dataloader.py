import torch
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms

from datasets import load_dataset

to_tensor = transforms.ToTensor()
resize = transforms.Resize((80, 80), antialias=True)


def preprocess(image):
    """convert a PIL Image to a pytorch Tensor of shape 1 * 3 * Height * Width"""
    return resize(to_tensor(image)).unsqueeze(0)


def collate_fn(batch):
    """
    :param batch:
    :return:
        images: Tensor of shape N * 3 * 80 * 80
        labels: Tensor of shape N * 1
    """
    # collect images and labels
    images = []
    labels = []
    for data in batch:
        images.append(preprocess(data["image"]))
        labels.append(data["labels"])

    # build tensor
    images = torch.concat(images, dim=0)
    labels = torch.Tensor(labels)
    return images, labels.unsqueeze(-1)


def get_dataset(mode="train"):
    dataset = load_dataset("keremberke/chest-xray-classification", name="full", verification_mode="no_checks")
    return dataset[mode]


def get_dataloader(batch_size=8, mode="train"):
    dataset = get_dataset(mode)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, drop_last=True, collate_fn=collate_fn)
    return dataloader


if __name__ == "__main__":
    dataloader = get_dataloader()
    for images, labels in dataloader:
        print(f"images shape: {images.shape}")  # 8 * 3 * 80 * 80
        print(f"labels shape: {labels.shape}")  # 8 * 1
        break
