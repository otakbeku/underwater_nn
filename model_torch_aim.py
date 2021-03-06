# import imgaug
import torch
import torch.nn as nn
from torchvision import transforms
import torch.optim as optim
# import torchvision
# import torchvision.models as models
from torch.utils.data.sampler import SubsetRandomSampler
from kornia import losses

from time import time
import random
import numpy as np

from aim import Run

from model_torch import *

# ================================== Utils
import gc
import os

SEED = 303

uwcnn_run = Run(
    repo='',
    experiment='UWCNN'
)

MODEL_CONFIG = {
    'epochs': 1000,
    'seed': 101,
    'train_folder':"data/EUVP/underwater_dark/trainA",
    'target_folder':"data/EUVP/underwater_dark/trainB",
    'lr':0.001,
    'weight_decay': 1e-05,
    'root_path': "",
    'batch_size': 4,
    'model_name':'UWCNN'
}
uwcnn_run['MODEL_CONFIG'] = MODEL_CONFIG

# gc.collect()
# torch.cuda.empty_cache()

device = "cpu"
if torch.cuda.is_available():
    device = "cuda:0"


os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


def set_seed(seed):
    # comment this if you have a problem with cudnn
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # imgaug.seed(seed)


def data_loader(train_folder, target_folder, batch_size=1, test_size=0.2, valid_size=0.2):
    transformer = transforms.Compose([transforms.Resize((230, 310)),transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
    dataset = Paired_Dataset(train_folder, target_folder, transform=transformer)
    dataset_size = len(dataset)
    indices = list(range(dataset_size))

    np.random.shuffle(indices)
    test_split = int(np.floor(test_size * dataset_size))
    train_indices = indices[test_split:]
    test_indices = indices[:test_split]

    train_size = len(train_indices)
    valid_split = int(np.floor((1 - valid_size) * train_size))
    train_indices, valid_indices = train_indices[:
                                                 valid_split], train_indices[valid_split:]

    valid_sampler = SubsetRandomSampler(valid_indices)
    valid_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=valid_sampler,
    )


    train_sampler = SubsetRandomSampler(train_indices)
    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=train_sampler,
    )

    test_sampler = SubsetRandomSampler(test_indices)
    test_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=test_sampler,
        num_workers=2
    )
    return train_loader, valid_loader, test_loader

def mse_ssim(output, target):
    # loss func:
    # MSE + SSIM
    mse_loss = nn.MSELoss()(output, target)
    ssim_loss = losses.ssim_loss(output, target, window_size=13)
    loss = mse_loss+ssim_loss
    return loss

model = T_CNN()
set_seed(MODEL_CONFIG['seed'])
optimizer = optim.SGD(model.parameters(), lr=MODEL_CONFIG['lr'], weight_decay=MODEL_CONFIG['weight_decay'])
train_loader, valid_loader, test_loader = data_loader(MODEL_CONFIG['train_folder'],MODEL_CONFIG['target_folder'], batch_size=MODEL_CONFIG['batch_size'])

# criterion = mse_ssim()
validation_loss_min = np.inf
# t_start = time()

model.to(device)
print('[INFO] Begin training')
for epoch in range(MODEL_CONFIG.get('epochs')):
    # print(f'Epoch: {epoch}')
    train_loss = 0.0
    validation_loss = 0.0
    model.train()
    for i, (data, target) in enumerate(train_loader):
        # print(f'\ttraining batch: {i}')
        data = data.to(device)
        target = target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = mse_ssim(output, target)

        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        
    train_loss = train_loss / len(train_loader.sampler)
    uwcnn_run.track(
        train_loss,
        name='Training loss',
        epoch=epoch,
        context={
            'subset': 'train'
        }
    )

        
    model.eval()
    for i, (data, target) in enumerate(valid_loader):
        # print(f'\tvalidation batch: {i}')
        data = data.to(device)
        target = target.to(device)

        output = model(data)
        loss = mse_ssim(output, target)
        validation_loss += loss.item()
        
    validation_loss = validation_loss / len(valid_loader.sampler)
    if validation_loss <= validation_loss_min:
        print(f'\t\tSmaller validation loss! {validation_loss_min} --> {validation_loss}')
        path = os.path.join(MODEL_CONFIG['root_path'], "checkpoint_best")
        torch.save((model.state_dict(), optimizer.state_dict()), path+f"/{MODEL_CONFIG['model_name']}_best.pt")
        validation_loss_min = validation_loss
    uwcnn_run.track(
        validation_loss,
        name='Validation loss',
        epoch=epoch,
        context={
            'subset': 'validation'
        }
    )
    if epoch %50==0:
        gc.collect()
    
    print(f'\tEPOCH: {epoch+1}\ttrain loss: {train_loss}\tValidation loss: {validation_loss}')
torch.save(model.state_dict(), f"{MODEL_CONFIG['model_name']}last_epoch_model.pt")
