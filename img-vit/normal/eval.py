from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import torch
from PIL import Image
from utils.metrics import make_evaluate
from utils.flickr8k import Flickr8kDataset
from torch.utils.data import DataLoader
import os
from tqdm import tqdm 


DATASET_BASE_PATH = './datasets/flickr8k/'

# load models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

feature_extractor = ViTImageProcessor.from_pretrained("atasoglu/vit-gpt2-flickr8k")
tokenizer = AutoTokenizer.from_pretrained("atasoglu/vit-gpt2-flickr8k")

model = VisionEncoderDecoderModel.from_pretrained("atasoglu/vit-gpt2-flickr8k")

model.to(device)
model.eval()

word2idx = tokenizer.get_vocab()
idx2word = {idx: word for word, idx in word2idx.items()}



# %%
def evaluate_model(data_loader, feature_extractor, tokenizer, model, idx2word, word2idx):

    references = []
    hypotheses = []
    for batch_idx, batch in enumerate(tqdm(data_loader)):
        img, captions = batch
        captions =  [int(c) for c in captions]
        with torch.no_grad():
            #     encoded_image = encoder(img).last_hidden_state
            #     decoder_outputs = decoder.generate( encoded_image,decoder_start_token_id=model.config.decoder.bos_token_id, max_length=800)
            #     print(decoder_outputs.shape)

            output = model.generate(img).squeeze(0)

            preds = tokenizer.batch_decode(output, skip_special_tokens=True)
            captions = tokenizer.batch_decode(captions, skip_special_tokens=True)


            # remove empty strings, symbol, convert to lowercase
            preds = [ p.lower() for p in preds if p not in ['','.',',','!','?']]

        references.append([captions])
        hypotheses.append(preds)

    bleu4 = make_evaluate(references, hypotheses, tokenizer)
    return bleu4


for IMAGE_TYPE in ['R0.5S50','R0.5S1','R0.2S50','R0.2S1','R0.1S50','R0.1S1','R1S1_GF500','R0.5S1_GF500']:
    print("Evaluating:",IMAGE_TYPE)


    eval_set = Flickr8kDataset(dataset_base_path=DATASET_BASE_PATH, image_type=IMAGE_TYPE,dist='test',tokenizer=tokenizer,feature_extractor=feature_extractor,  device=device)
    
    eval_loader = DataLoader(eval_set, batch_size=1, shuffle=False, sampler=None, pin_memory=False)
    
    with torch.no_grad():
        model.eval()
        eval_bleu = evaluate_model(data_loader=eval_loader,
                                      feature_extractor=feature_extractor,
                                      tokenizer=tokenizer,
                                      model=model,
                                      idx2word = idx2word,
                                      word2idx= word2idx)
        print(f'BLEU-4 {IMAGE_TYPE}: {eval_bleu}')
    print("========================================\n\n\n")