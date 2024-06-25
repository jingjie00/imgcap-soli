import os
import sys
sys.path.append(os.path.abspath('..'))
import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from utils.SiameseCaptionDataset import SiameseCaptionDataset
from SiameseNetwork import SiameseNetwork, ContrastiveLoss
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
from PIL import Image
import random
from tqdm import tqdm



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATASET_BASE_PATH = './../datasets/flickr8k/'

MODEL_NAME = "vit_siamese"
NUM_EPOCHS = 50
LOG_INTERVAL = 10

feature_extractor = ViTImageProcessor.from_pretrained("atasoglu/vit-gpt2-flickr8k")
tokenizer = AutoTokenizer.from_pretrained("atasoglu/vit-gpt2-flickr8k")
vit = VisionEncoderDecoderModel.from_pretrained("atasoglu/vit-gpt2-flickr8k")
train_set = SiameseCaptionDataset(dataset_base_path=DATASET_BASE_PATH,feature_extractor=feature_extractor, tokenizer=tokenizer,device=device, dist='train')
val_set = SiameseCaptionDataset(dataset_base_path=DATASET_BASE_PATH, feature_extractor=feature_extractor, tokenizer=tokenizer, device=device, dist='test')


def train_model(train_loader, model, optimizer, contrastive_loss, cross_entropy_loss):
    running_acc = 0.0
    running_loss = 0.0
    model.train()
    for batch_idx, batch in enumerate(tqdm(train_loader)):
        
        optimizer.zero_grad()
        anchor,pair, s_label = batch
        images1, captions1, caplen1 = anchor
        images2, captions2, caplen2 = pair

        output1 = model(images1, captions1)
        output2 = model(images2, captions2)

        labels1 = captions1[:, 1:].contiguous().view(-1)  # Shift the labels for causal language modeling
        labels2 = captions2[:, 1:].contiguous().view(-1)

        logits1 = output1.logits[:, :-1].contiguous().view(-1, output1.logits.size(-1))
        logits2 = output2.logits[:, :-1].contiguous().view(-1, output2.logits.size(-1))

        emb1 = model.encoder(images1).last_hidden_state
        emb2 = model.encoder(images2).last_hidden_state

        
        celoss1 = cross_entropy_loss(logits1, labels1)
        celoss2 = cross_entropy_loss(logits2, labels2)
        ctloss = contrastive_loss(emb1,emb2, s_label)

        loss = celoss1 + celoss2 + ctloss

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        predictions1 = torch.argmax(output1.logits, dim=-1)
        predictions2 = torch.argmax(output2.logits, dim=-1)
        
        correct_predictions1 = (predictions1[:, :-1] == captions1[:, 1:]).float().sum()
        correct_predictions2 = (predictions2[:, :-1] == captions2[:, 1:]).float().sum()
        
        accuracy1 = correct_predictions1 / (captions1[:, 1:].ne(tokenizer.pad_token_id).sum().float())
        accuracy2 = correct_predictions2 / (captions2[:, 1:].ne(tokenizer.pad_token_id).sum().float())
        
        running_acc += (accuracy1.item() + accuracy2.item()) / 2

        
        if (batch_idx + 1) % LOG_INTERVAL == 0:
            print(f'Batch {(batch_idx+1)}/{len(train_loader)} \t| train_loss: {running_loss / (batch_idx + 1):.4f} train_acc: {running_acc / (batch_idx + 1):.4f}')


    return running_loss / len(train_loader)

def validate_model(val_loader, model, contrastive_loss, cross_entropy_loss):
    model.eval()
    running_val_loss = 0.0
    running_val_acc = 0.0

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(val_loader, desc="Validation")):
            anchor, pair, s_label = batch
            images1, captions1, caplen1 = anchor
            images2, captions2, caplen2 = pair
            
            images1, images2 = images1.to(device), images2.to(device)
            captions1, captions2 = captions1.to(device), captions2.to(device)
            s_label = s_label.to(device)

            output1 = model(images1, captions1)
            output2 = model(images2, captions2)

            labels1 = captions1[:, 1:].contiguous().view(-1)
            labels2 = captions2[:, 1:].contiguous().view(-1)

            logits1 = output1.logits[:, :-1].contiguous().view(-1, output1.logits.size(-1))
            logits2 = output2.logits[:, :-1].contiguous().view(-1, output2.logits.size(-1))

            emb1 = model.encoder(images1).last_hidden_state
            emb2 = model.encoder(images2).last_hidden_state

            celoss1 = cross_entropy_loss(logits1, labels1)
            celoss2 = cross_entropy_loss(logits2, labels2)
            ctloss = contrastive_loss(emb1, emb2, s_label)

            val_loss = celoss1 + celoss2 + ctloss
            running_val_loss += val_loss.item()

            predictions1 = torch.argmax(output1.logits, dim=-1)
            predictions2 = torch.argmax(output2.logits, dim=-1)

            correct_predictions1 = (predictions1[:, :-1] == captions1[:, 1:]).float().sum()
            correct_predictions2 = (predictions2[:, :-1] == captions2[:, 1:]).float().sum()

            accuracy1 = correct_predictions1 / (captions1[:, 1:].ne(tokenizer.pad_token_id).sum().float())
            accuracy2 = correct_predictions2 / (captions2[:, 1:].ne(tokenizer.pad_token_id).sum().float())

            running_val_acc += (accuracy1.item() + accuracy2.item()) / 2

    avg_val_loss = running_val_loss / len(val_loader)
    avg_val_acc = running_val_acc / len(val_loader)
    print(f'Validation Loss: {avg_val_loss:.4f} Validation Accuracy: {avg_val_acc:.4f}')
    return avg_val_loss

    
vit = vit.to(device)
contrastive_loss = ContrastiveLoss().to(device)
cross_entropy_loss =nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(vit.parameters(), lr=0.001)


train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
val_loader = DataLoader(val_set, batch_size=16, shuffle=False)

val_loss_min = float('inf')
train_loss_min = float('inf')

for epoch in range(NUM_EPOCHS):
    print(f'Epoch {epoch + 1}/{NUM_EPOCHS}')
    train_loss = train_model(model=vit, optimizer=optimizer,
                             train_loader=train_loader,
                             contrastive_loss=contrastive_loss,
                             cross_entropy_loss=cross_entropy_loss)

    with torch.no_grad():
        val_loss = validate_model(model=vit, 
                                  val_loader=val_loader,
                                  contrastive_loss=contrastive_loss,
                                 cross_entropy_loss=cross_entropy_loss)
        print(f'Training Loss: {train_loss:.4f}')
        print(f'Validation Loss: {val_loss:.4f}')
        
        state = {
            'epoch': epoch + 1,
            'state_dict': vit.state_dict(),
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


