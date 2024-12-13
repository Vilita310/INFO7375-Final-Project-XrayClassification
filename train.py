import torch
from torch.nn import BCELoss

from tqdm import tqdm

from xray_classification.data.dataloader import get_dataloader
from xray_classification.model.model import XRayClassifier


def main(n_epochs=3):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # load data
    training_loader = get_dataloader()

    # load model
    model = XRayClassifier()
    model.to(device)

    # load optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # loss function
    loss_fn = BCELoss()

    # start training
    for cur_epoch in range(n_epochs):
        running_loss = 0.

        for i, (images, labels) in enumerate(tqdm(training_loader)):
            images, labels = images.to(device), labels.to(device)

            # Zero your gradients for every batch!
            optimizer.zero_grad()

            # Make predictions for this batch
            outputs = model(images)

            # Compute the loss and its gradients
            loss = loss_fn(outputs, labels)
            loss.backward()

            # Adjust learning weights
            optimizer.step()

            # Gather data and report
            running_loss += loss.item()

        last_avg_loss = running_loss / (i + 1)  # loss per batch
        print(f'average loss for epoch {cur_epoch}: {last_avg_loss}')

        # save model
        torch.save(model, f"model_checkpoint_{cur_epoch}.pth", )


if __name__ == "__main__":
    main()
