import glob
import io
import ntpath
import os
from transformers import AutoTokenizer, ViTImageProcessor
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from .utils_torch import split_data
import random


class SiameseCaptionDataset(Dataset):
    """
    imgname: just image file name
    imgpath: full path to image file
    """

    def __init__(self, dataset_base_path='../datasets/flickr8k/',
                 image_type="normal", dist='val',
                 tokenizer = None,
                 feature_extractor = None,
                 device=torch.device('cpu')):
        self.resolutions = ['flickr8k_images', 'R0.1S1', 'R0.1S50', 'R0.2S1', 'R0.2S50', 'R0.5S1', 'R0.5S50', 'R0.05S50', 'R0.5S1_GF500','R1S1_GF500']
        self.token = dataset_base_path + 'Flickr8k.token.txt'

        self.dist_list = {
            'train': dataset_base_path + 'Flickr_8k.trainImages.txt',
            'val': dataset_base_path + 'Flickr_8k.devImages.txt',
            'test': dataset_base_path + 'Flickr_8k.testImages.txt'
        }

        self.pil_d = None

        self.device = torch.device(device)
        self.torch = torch.cuda if (self.device.type == 'cuda') else torch

        self.dataset_base_path = dataset_base_path
        self.images_path = dataset_base_path+'flickr8k_images/'

        self.imgpath_list = glob.glob(self.images_path + '*.jpg')
        self.all_imgname_to_caplist = self.__all_imgname_to_caplist_dict()
        self.imgname_to_caplist = self.__get_imgname_to_caplist_dict(self.__get_imgpath_list(dist=dist))

        self.db = self.get_db()

        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor
        self.max_len = 50




    def get_db(self):
        # ----- Forming a df to sample from ------
        l = ["image_id\tcaption\n"]

        for imgname, caplist in self.imgname_to_caplist.items():
            for cap in caplist:
                l.append(
                    f"{imgname}\t"
                    f"{cap.lower()}\n")
        img_id_cap_str = ''.join(l)

        df = pd.read_csv(io.StringIO(img_id_cap_str), delimiter='\t')
        return df.to_numpy()

    
    @property
    def pad_value(self):
        return 0
    
    def __all_imgname_to_caplist_dict(self):
        captions = open(self.token, 'r').read().strip().split('\n')
        imgname_to_caplist = {}
        for i, row in enumerate(captions):
            row = row.split('\t')
            row[0] = row[0][:len(row[0]) - 2]  # filename#0 caption
            if row[0] in imgname_to_caplist:
                imgname_to_caplist[row[0]].append(row[1])
            else:
                imgname_to_caplist[row[0]] = [row[1]]
        return imgname_to_caplist
    
    def __get_imgname_to_caplist_dict(self, img_path_list):
        d = {}
        for i in img_path_list:
            if i[len(self.images_path):] in self.all_imgname_to_caplist:
                d[ntpath.basename(i)] = self.all_imgname_to_caplist[i[len(self.images_path):]]
        return d

    def __get_imgpath_list(self, dist='val'):
        dist_images = set(open(self.dist_list[dist], 'r').read().strip().split('\n'))
        dist_imgpathlist = split_data(dist_images, img=self.imgpath_list, images=self.images_path)
        return dist_imgpathlist
    
    def __getitem__(self, index: int):
        def get(index, res):
            imgname = self.db[index][0]
            caption = self.db[index][1]
            
            cap_toks = self.tokenizer.tokenize(caption)
            cap_toks = cap_toks[:-1]
            cap_toks = self.tokenizer.convert_tokens_to_ids(cap_toks)

            cap_len = len(cap_toks)

            if cap_len < self.max_len:
                cap_toks += [self.tokenizer.pad_token_id] * (self.max_len - cap_len)

            cap_toks = torch.LongTensor(cap_toks).to(self.device)
            
            img = Image.open(os.path.join(self.dataset_base_path, res, imgname))
            if img.size[0] == 1:
                img = transforms.Resize((2,img.size[1]))(img)
            
            if img.size[1] == 1:
                img = transforms.Resize((img.size[0], 2))(img)

            img = img.convert('RGB')
            img = self.feature_extractor(images=img, return_tensors="pt").pixel_values
            img = img.squeeze(0)
            img = img.to(self.device)
            
            return img, cap_toks, cap_len

        res1 = random.choice(self.resolutions)
        resolutions_temp = self.resolutions.copy()
        resolutions_temp.remove(res1)
        res2 = random.choice(resolutions_temp)
        
        anchor = get(index, res1)
        if random.random() < 0.5:
            pair = get(index, res2)
            s_label = torch.FloatTensor([0]).to(self.device)
        else:
            negative_idx = random.randint(0, len(self.db) - 1)
            while negative_idx == index:
                negative_idx = random.randint(0, len(self.db) - 1)
            pair = get(negative_idx, res2)
            s_label = torch.FloatTensor([1]).to(self.device)
        
        return anchor, pair, s_label

    def __len__(self):
        return len(self.db)

    def get_image_captions(self, index: int):
        """
        :param index: [] index
        :returns: image_path, list_of_captions
        """
        imgname = self.db[index][0]
        return os.path.join(self.images_path, imgname), self.imgname_to_caplist[imgname]



# if main
if __name__ == '__main__':
    dataset = SiameseCaptionDataset(tokenizer=AutoTokenizer.from_pretrained("atasoglu/vit-gpt2-flickr8k"),
                                    feature_extractor=ViTImageProcessor.from_pretrained("atasoglu/vit-gpt2-flickr8k"))
    print(dataset.get_db())
    print(dataset.get_image_captions(0))
    print(dataset.__getitem__(0))
    print(dataset.__len__())
