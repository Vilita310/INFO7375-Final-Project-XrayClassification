import os
import random

import cv2
import numpy as np
import torch
from PIL import Image

from xray_classification.data.dataloader import get_dataset, preprocess

checkpoint_file = "model_checkpoint_2.pth"
image_folder = "/Users/caojing/Documents/xray_patients"
diagnose_folder = "/Users/caojing/Documents/xray_diagnoses"


def get_one_image():
    dataset = get_dataset(mode="test")
    data = random.choice(dataset)
    return data["image"]


def infer():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # option 1: pick one image from test set
    #image = get_one_image()
    #image.save(os.path.join(image_folder, "xray_51119.jpg"))
    #return


    # option 2: load the first image from `image_folder`
    img_name = os.listdir(image_folder)[0]
    img_path = os.path.join(image_folder, img_name)

    image = Image.open(img_path)

    # if we want to process multiple images, we can use a for loop
    # for img_name in os.listdir(image_folder):
    #     img_path = os.path.join(image_folder, img_name)
    #     image = Image.open(img_path)
    

    # load model, load weights and additional information (such as optimizer status, training status, etc.).
    model = torch.load(checkpoint_file, map_location=device, weights_only=False)

    # preprocess image
    inputs = preprocess(image)
    inputs = inputs.to(device)

    # forward
    probs = model(inputs)

    # output
    img = np.array(image)
    text = f"Probability of pneumonia: {probs.item():.0%}."
    cv2.putText(img, text, (0, 610), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, color=(255, 0, 0), thickness=2)
    cv2.imshow(img_path, img)
    cv2.waitKey(-1)

    target_path = os.path.join(diagnose_folder, f"{img_name[:-4]}_pneumonia_{probs.item():.0%}.jpg")
    cv2.imwrite(target_path, img)
    print(f"Diagnose result saved to {target_path}.")


if __name__ == "__main__":
    infer()
