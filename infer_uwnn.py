import torch
import os
from model_torch import Infer_Dataset, T_CNN
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
import cv2
import matplotlib.pyplot as plt
import numpy as np

SEED = 303

MODEL_CONFIG = {
    'test_folder': "/home/Downloads/underwater_dark/test/",
    'model_name': 'UWCNN'
}


def data_loader(batch_size=1):
    transformer = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    dataset = Infer_Dataset(MODEL_CONFIG["test_folder"], transform=transformer)
    dataset_size = len(dataset)
    print(f"test set data size {dataset_size}")

    test_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=2
    )

    return test_loader


device = "cpu"
if torch.cuda.is_available():
    device = "cuda:0"

model = T_CNN()
model.load_state_dict(torch.load("checkpoint_best/UWCNN_best.pt")[0])
model.to(device)
model.eval()

test_loader = data_loader(1)

counter = 0
for data in test_loader:
    counter += 1
    print(f"infer: {data[1]}")
    data = data[0].to(device)
    with torch.no_grad():
        output = model.forward(data)
        MEAN = torch.tensor([0.485, 0.456, 0.406])
        STD = torch.tensor([0.229, 0.224, 0.225])
        x = (output.cpu() * STD[:, None, None] + MEAN[:, None, None])
        x = x.numpy().squeeze().transpose([1, 2, 0])

    print(x)
    print(x.shape)
    write_x = (x * 255).astype("uint8")
    write_x = cv2.cvtColor(write_x, cv2.COLOR_BGR2RGB)

    print(write_x)
    print(write_x.shape)
    cv2.imwrite(f"image{counter}.png", write_x)
    fig = plt.Figure()
    plt.imshow(x)
    plt.show()

