# %%
import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(parent_dir)

import pickle
from matplotlib import pyplot as plt
import tqdm
import torch
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader
from torchvision import transforms

from utils.flickr8k import Flickr8kDataset
from utils.glove import embedding_matrix_creator

from utils.utils_torch import words_from_tensors_fn
from utils.metrics import accuracy_fn, make_evaluate


from models.torch.resnet101_attention import Captioner
# %%
device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
DATASET_BASE_PATH = './../datasets/flickr8k/'

# %%

train_transformations = transforms.Compose([
    transforms.Resize(256),  # smaller edge of image resized to 256
    transforms.RandomCrop(224), 
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),  # convert the PIL Image to a tensor
    transforms.Normalize((0.485, 0.456, 0.406),  # normalize image for pre-trained model
                         (0.229, 0.224, 0.225))
])
eval_transformations = transforms.Compose([
    transforms.Resize(256),  # smaller edge of image resized to 256
    transforms.RandomCrop(224), 
    transforms.ToTensor(),  # convert the PIL Image to a tensor
    transforms.Normalize((0.485, 0.456, 0.406),  # normalize image for pre-trained model
                         (0.229, 0.224, 0.225))
])

# %%
MODEL = "resnet101_attention"
EMBEDDING_DIM = 300
EMBEDDING = f"GLV{EMBEDDING_DIM}"
HIDDEN_SIZE = 256
BATCH_SIZE = 128
LR = 1e-2
ATTENTION_DIM = 256
DECODER_SIZE = 256
MODEL_NAME = f'./saved/{MODEL}_b{BATCH_SIZE}_emd{EMBEDDING}_att{ATTENTION_DIM}dec{DECODER_SIZE}'
NUM_EPOCHS = 50
LOG_INTERVAL = 100



# %%

def train_model(train_loader, model, loss_fn, optimizer):
    running_acc = 0.0
    running_loss = 0.0
    model.train()
    for batch_idx, batch in enumerate(train_loader):
        images, captions, lengths = batch
        sort_ind = torch.argsort(lengths, descending=True)
        images = images[sort_ind]
        captions = captions[sort_ind]
        lengths = lengths[sort_ind]

        optimizer.zero_grad()
        scores, caps_sorted, decode_lengths, alphas, sort_ind = model(images, captions, lengths)
        targets = caps_sorted[:, 1:]

        scores = pack_padded_sequence(scores, decode_lengths, batch_first=True)[0]
        targets = pack_padded_sequence(targets, decode_lengths, batch_first=True)[0]

        loss = loss_fn(scores, targets)
        loss.backward()
        optimizer.step()

        running_acc += (torch.argmax(scores, dim=1) == targets).sum().float().item() / targets.size(0)
        running_loss += loss.item()

        if (batch_idx + 1) % LOG_INTERVAL == 0:
            print(f'Batch {(batch_idx+1)}/{len(train_loader)} \t| train_loss: {running_loss / (batch_idx + 1):.4f} train_acc: {running_acc / (batch_idx + 1):.4f}')

    return running_loss / len(train_loader)

def validate_model(data_loader, model, loss_fn):
    model.eval()
    running_acc=0.0
    running_loss=0.0
    for batch_idx, batch in enumerate(data_loader):
        images, captions, lengths = batch
        sort_ind = torch.argsort(lengths, descending=True)
        images = images[sort_ind]
        captions = captions[sort_ind]
        lengths = lengths[sort_ind]

        scores, caps_sorted, decode_lengths, alphas, sort_ind = model(images, captions, lengths)
        targets = caps_sorted[:, 1:]

        scores = pack_padded_sequence(scores, decode_lengths, batch_first=True)[0]
        targets = pack_padded_sequence(targets, decode_lengths, batch_first=True)[0]

        loss = loss_fn(scores, targets)

        running_acc += (torch.argmax(scores, dim=1) == targets).sum().float().item() / targets.size(0)
        running_loss += loss.item()

        if (batch_idx + 1) % LOG_INTERVAL == 0:
            print(f'Batch {(batch_idx+1)}/{len(train_loader)} \t| val_loss: {running_loss / (batch_idx + 1):.4f} val_acc: {running_acc / (batch_idx + 1):.4f}')

        return running_loss / len(data_loader)
# %%
def evaluate_model(data_loader, model, idx2word, word2idx):
    references = []
    hypotheses = []
    for batch_idx, batch in enumerate(data_loader):
        images, captions, lengths = batch
        outputs = tensor_to_word_fn(model.sample(images,word2idx['<start>']).cpu().numpy())
        captions = tensor_to_word_fn(torch.stack(captions).cpu().numpy())
        captions = [[cap] for cap in captions]
        references.extend(captions)
        hypotheses.extend(outputs)
    bleu4 = make_evaluate(references, hypotheses, idx2word, word2idx)

    print("Sample references:", references[0],references[5],references[10])
    print("Sample hypotheses:", hypotheses[0],hypotheses[5],hypotheses[10])
    return bleu4

#%%
def show_sample(model, dset, i):
    i = 0
    im, cp, _ = dset[i]
    model.eval()
    sampled_caption = model.sample(im.unsqueeze(0), word2idx['<start>'])[0]
    caption_text = ''.join([idx2word[idx.item()] + ' ' for idx in sampled_caption])
    print(caption_text)
    print(dset.get_image_captions(i)[1])

# %%
train_set = Flickr8kDataset(dataset_base_path=DATASET_BASE_PATH, dist='train', device=device,
                            return_type='tensor',
                            load_img_to_memory=False)

vocab_set = train_set.get_vocab()
vocab, word2idx, idx2word, max_len = vocab_set
vocab_size = len(vocab)
vocab_size, max_len

val_set = Flickr8kDataset(dataset_base_path=DATASET_BASE_PATH, dist='val', vocab_set=vocab_set, device=device,
                          return_type='tensor',
                          load_img_to_memory=False)

with open('./saved/vocab_set.pkl', 'wb') as f:
    pickle.dump(train_set.get_vocab(), f)
len(train_set), len(val_set)



# %%
embedding_matrix = embedding_matrix_creator(embedding_dim=EMBEDDING_DIM, word2idx=word2idx)
embedding_matrix.shape



# %%
model = Captioner(encoded_image_size=14, encoder_dim=2048,
                        attention_dim=ATTENTION_DIM, embed_dim=EMBEDDING_DIM, decoder_dim=DECODER_SIZE,
                        vocab_size=vocab_size,
                        embedding_matrix=embedding_matrix, train_embd=False).to(device)

loss_fn = torch.nn.CrossEntropyLoss(ignore_index=train_set.pad_value).to(device)
acc_fn = accuracy_fn(ignore_value=train_set.pad_value)
tensor_to_word_fn = words_from_tensors_fn(idx2word=idx2word)

params = model.parameters()

optimizer = torch.optim.RMSprop(params=params, lr=LR)
# %%


train_set.transformations = train_transformations
val_set.transformations = eval_transformations
# %%
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, sampler=None, pin_memory=False)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, sampler=None, pin_memory=False)

# %%
eval_collate_fn = lambda batch: (torch.stack([x[0] for x in batch]), [x[1] for x in batch], [x[2] for x in batch])
evaluate_val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, sampler=None, pin_memory=False,
                        collate_fn=eval_collate_fn)



# %%
val_loss_min = float('inf')
train_loss_min = float('inf')
val_bleu4_max = 0.0
for epoch in range(NUM_EPOCHS):
    print(f'Epoch {epoch + 1}/{NUM_EPOCHS}')
    train_loss = train_model(model=model,
                             optimizer=optimizer, loss_fn=loss_fn, 
                             train_loader=train_loader)
    train_loss=0
    with torch.no_grad():
        val_loss = validate_model(model=model, 
                                  loss_fn=loss_fn, 
                                  data_loader=val_loader)
        
        val_bleu4 = evaluate_model(model=model,
                                  idx2word = idx2word,
                                  word2idx = word2idx,
                                  data_loader=evaluate_val_loader)
        
        state = {
            'epoch': epoch + 1,
            'embed_size': EMBEDDING_DIM,
            'hidden_size': HIDDEN_SIZE,
            'vocab_size': vocab_size,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'train_loss_latest': train_loss,
            'val_loss_latest': val_loss,
            'val_bleu4_latest': val_bleu4,
            'train_loss_min': min(train_loss, train_loss_min),
            'val_loss_min': min(val_loss, val_loss_min),
            'val_bleu4_max': max(val_bleu4, val_bleu4_max)
        }
        if val_loss_min > val_loss:
            val_loss = val_loss_min
            torch.save(state, f'{MODEL_NAME}''_best_val_loss.pt')
        if val_bleu4 > val_bleu4_max:
            val_bleu4_max = val_bleu4
            torch.save(state, f'{MODEL_NAME}''_best_val_bleu4.pt')

        show_sample(model, train_set, 0)
        show_sample(model, val_set, 0)


torch.save(state, f'{MODEL_NAME}_ep{NUM_EPOCHS:02d}_weights.pt')
