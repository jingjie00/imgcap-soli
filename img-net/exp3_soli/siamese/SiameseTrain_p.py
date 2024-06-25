import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from SiameseDataset_p import SiameseDataset
from SiameseNetwork import SiameseNetwork, ContrastiveLoss, Encoder

from PIL import Image
import random
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATASET_BASE_PATH = './../../datasets/flickr8k/'
EMBEDDING_DIM = 50

MODEL_NAME = "resnet50_siamese"
NUM_EPOCHS = 50
LOG_INTERVAL = 10


train_transformations = transforms.Compose([
    transforms.Resize(256),  # smaller edge of image resized to 256
    transforms.RandomCrop(224), 
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),  # convert the PIL Image to a tensor
    transforms.Normalize((0.485, 0.456, 0.406),  # normalize image for pre-trained model
                         (0.229, 0.224, 0.225))
])

train_set = SiameseDataset(base_dir=DATASET_BASE_PATH, transform=train_transformations)
val_set = SiameseDataset(base_dir=DATASET_BASE_PATH, transform=train_transformations)


def train_model(train_loader, model, loss_fn, optimizer):
    running_loss = 0.0
    model.train()
    for batch_idx, batch in enumerate(train_loader):
        optimizer.zero_grad()
        image1, image2 = batch[0]
        targets = batch[1]

        output1, output2 = model(image1.to(device), image2.to(device))
        loss = loss_fn(output1, output2, targets.to(device))
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if (batch_idx + 1) % LOG_INTERVAL == 0:
            print(f'Batch {(batch_idx+1)}/{len(train_loader)} \t| train_loss: {running_loss / (batch_idx + 1):.4f}')

    return running_loss / len(train_loader)

def validate_model(val_loader, model, loss_fn):
    running_loss = 0
    model.eval()
    for batch_idx, batch in enumerate(val_loader):
        image1, image2 = batch[0]
        targets = batch[1]

        output1, output2 = model(image1.to(device), image2.to(device))
        loss = loss_fn(output1, output2, targets.to(device))

        running_loss += loss.item()

    return running_loss / len(val_loader)

    
encoder = Encoder(embed_size=EMBEDDING_DIM).to(device)
model = SiameseNetwork(encoder).to(device)
loss_fn = ContrastiveLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
val_loader = DataLoader(val_set, batch_size=32, shuffle=False)

val_loss_min = float('inf')
train_loss_min = float('inf')

for epoch in range(NUM_EPOCHS):
    print(f'Epoch {epoch + 1}/{NUM_EPOCHS}')
    train_loss = train_model(model=model,
                             optimizer=optimizer, loss_fn=loss_fn, 
                             train_loader=train_loader)
    with torch.no_grad():
        val_loss = validate_model(model=model, 
                                  loss_fn=loss_fn, 
                                  val_loader=val_loader)
        print(f'Training Loss: {train_loss:.4f}')
        print(f'Validation Loss: {val_loss:.4f}')
        
        state = {
            'epoch': epoch + 1,
            'embed_size': EMBEDDING_DIM,
            'state_dict': model.encoder.state_dict(),
            'optimizer': optimizer.state_dict(),
            'train_loss_latest': train_loss,
            'val_loss_latest': val_loss,
            'train_loss_min': min(train_loss, train_loss_min),
            'val_loss_min': min(val_loss, val_loss_min)
        }
        if val_loss_min > val_loss:
            val_loss_min = val_loss
            torch.save(state, f'./saved/{MODEL_NAME}''_best_val_loss.pt')

torch.save(state, f'./saved/{MODEL_NAME}_ep{NUM_EPOCHS:02d}_weights.pt')


