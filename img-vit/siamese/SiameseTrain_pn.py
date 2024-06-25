import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from SiameseDataset_pn import SiameseDataset
from SiameseNetwork import SiameseNetwork, ContrastiveLoss
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
from PIL import Image
import random
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATASET_BASE_PATH = './../datasets/flickr8k/'

MODEL_NAME = "vit_siamese"
NUM_EPOCHS = 50
LOG_INTERVAL = 10

feature_extractor = ViTImageProcessor.from_pretrained("atasoglu/vit-gpt2-flickr8k")
vit = VisionEncoderDecoderModel.from_pretrained("atasoglu/vit-gpt2-flickr8k")
encoder = vit.encoder

train_set = SiameseDataset(base_dir=DATASET_BASE_PATH,feature_extractor=feature_extractor)
val_set = SiameseDataset(base_dir=DATASET_BASE_PATH, feature_extractor=feature_extractor)


def train_model(train_loader, model, loss_fn, optimizer):
    running_loss = 0.0
    model.train()
    for batch_idx, batch in enumerate(train_loader):
        optimizer.zero_grad()
        image1, image2 = batch[0]
        targets = batch[1]

        output1, output2 = siamese(image1.to(device), image2.to(device))
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

        output1, output2 = siamese(image1.to(device), image2.to(device))
        loss = loss_fn(output1, output2, targets.to(device))
        running_loss += loss.item()
    return running_loss / len(val_loader)

    
encoder = encoder.to(device)
siamese = SiameseNetwork(encoder).to(device)
loss_fn = ContrastiveLoss().to(device)
optimizer = torch.optim.Adam(siamese.parameters(), lr=0.001)


train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
val_loader = DataLoader(val_set, batch_size=32, shuffle=False)

val_loss_min = float('inf')
train_loss_min = float('inf')

for epoch in range(NUM_EPOCHS):
    print(f'Epoch {epoch + 1}/{NUM_EPOCHS}')
    train_loss = train_model(model=siamese,
                             optimizer=optimizer, loss_fn=loss_fn, 
                             train_loader=train_loader)
    with torch.no_grad():
        val_loss = validate_model(model=siamese, 
                                  loss_fn=loss_fn, 
                                  val_loader=val_loader)
        print(f'Training Loss: {train_loss:.4f}')
        print(f'Validation Loss: {val_loss:.4f}')
        
        state = {
            'epoch': epoch + 1,
            'state_dict': siamese.encoder.state_dict(),
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


