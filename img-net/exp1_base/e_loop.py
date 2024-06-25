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
from models.torch.densenet201_monolstm import Captioner

from utils.utils_torch import words_from_tensors_fn
from utils.metrics import accuracy_fn, make_evaluate
# %%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device

# %%
DATASET_BASE_PATH = './../datasets/flickr8k/'
eval_transformations = transforms.Compose([
    transforms.Resize(256),  # smaller edge of image resized to 256
    transforms.CenterCrop(224),  # get 224x224 crop from random location
    transforms.ToTensor(),  # convert the PIL Image to a tensor
    transforms.Normalize((0.485, 0.456, 0.406),  # normalize image for pre-trained model
                         (0.229, 0.224, 0.225))
])


# %%
vocab_set = pickle.load(open('./saved/vocab_set.pkl', 'rb'))
vocab, word2idx, idx2word, max_len = vocab_set
vocab_size = len(vocab)
tensor_to_word_fn = words_from_tensors_fn(idx2word=idx2word)



# %%
def evaluate_model(data_loader, model, idx2word,word2idx):

    references = []
    hypotheses = []
    for batch_idx, batch in enumerate(data_loader):
        images, captions, lengths = batch
        outputs = tensor_to_word_fn(model.sample(images,word2idx['<start>']).cpu().numpy())
        captions = tensor_to_word_fn(torch.stack(captions).cpu().numpy())
        captions = [[cap] for cap in captions]
        references.extend(captions)
        hypotheses.extend(outputs)
    bleu4 = make_evaluate(references, hypotheses,idx2word,word2idx)
    return bleu4


# %%
MODEL = "resnet50_monolstm"
EMBEDDING_DIM = 50
EMBEDDING = f"GLV{EMBEDDING_DIM}"
HIDDEN_SIZE = 256
BATCH_SIZE = 16
MODEL_NAME = f'./saved/{MODEL}_b{BATCH_SIZE}_emd{EMBEDDING}'
embedding_matrix = embedding_matrix_creator(embedding_dim=EMBEDDING_DIM, word2idx=word2idx)
embedding_matrix.shape


# %%
model = Captioner(EMBEDDING_DIM, HIDDEN_SIZE, vocab_size, num_layers=2,
                        embedding_matrix=embedding_matrix, train_embd=False).to(device)
checkpoint = torch.load(f'{MODEL_NAME}_ep50_weights.pt',map_location="cuda")
model.load_state_dict(checkpoint['state_dict'])


for IMAGE_TYPE in ['flickr8k_images','R0.5S50','R0.5S1','R0.2S50','R0.2S1','R0.1S50','R0.1S1','R1S1_GF500','R0.5S1_GF500']:
    print("Evaluating:",IMAGE_TYPE)


    eval_set = Flickr8kDataset(dataset_base_path=DATASET_BASE_PATH, image_type=IMAGE_TYPE,dist='test', vocab_set=vocab_set, device=device,
                          return_type='tensor',
                          load_img_to_memory=False)
    eval_set.transformations = eval_transformations
    eval_collate_fn = lambda batch: (torch.stack([x[0] for x in batch]), [x[1] for x in batch], [x[2] for x in batch])
    eval_loader = DataLoader(eval_set, batch_size=BATCH_SIZE, shuffle=False, sampler=None, pin_memory=False,
                            collate_fn=eval_collate_fn)

    with torch.no_grad():
        model.eval()
        eval_bleu = evaluate_model(data_loader=eval_loader,
                                      model=model,
                                      idx2word = idx2word, word2idx = word2idx)
        print(f'BLEU-4 {IMAGE_TYPE}: {eval_bleu}')
    print("========================================\n\n\n")