
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize
import nltk
import torch
from sacrebleu.metrics import BLEU as sacreBLEU
from .pymetrics import Metrics as pymetrics


# def corpus_meteor(s_s_reference_list, s_s_hypothesis_list):
#     assert len(s_s_reference_list) == len(s_s_hypothesis_list)
#     meteor_sum = 0
#     for reference_sentences, hypothesis in zip(s_s_reference_list,s_s_hypothesis_list):
#         score = meteor_score(reference_sentences, hypothesis)
#         meteor_sum += score
#     return meteor_sum / len(s_s_reference_list)
def accuracy_fn(ignore_value: int = 0):
    def accuracy_ignoring_value(source: torch.Tensor, target: torch.Tensor):
        mask = target != ignore_value
        return (source[mask] == target[mask]).sum().item() / mask.sum().item()

    return accuracy_ignoring_value

def corpus_meteor(s_s_reference_list, s_s_hypothesis_list):
    assert len(s_s_reference_list) == len(s_s_hypothesis_list)
    meteor_sum = 0
    for reference_sentences, hypothesis in zip(s_s_reference_list,s_s_hypothesis_list):
        score = meteor_score(reference_sentences, hypothesis)
        meteor_sum += score
    return meteor_sum / len(s_s_reference_list)

def make_evaluate(s_s_references_list, s_s_hypothesis_list, idx2word, word2idx):

    meteor = corpus_meteor(s_s_references_list, s_s_hypothesis_list)

    cs_references_list = [[' '.join(r) for r in rs]for rs in s_s_references_list]
    cs_hypothesis_list = [' '.join(h) for h in s_s_hypothesis_list]


    bleu_sacre_obj = sacreBLEU()
    bleu_sacre = bleu_sacre_obj.corpus_score(cs_hypothesis_list, cs_references_list)

    bleu_nltk_1 = corpus_bleu(cs_references_list, cs_hypothesis_list, weights=(1, 0, 0, 0))
    bleu_nltk_2 = corpus_bleu(cs_references_list, cs_hypothesis_list, weights=(0.5, 0.5, 0, 0))
    bleu_nltk_3 = corpus_bleu(cs_references_list, cs_hypothesis_list, weights=(0.33, 0.33, 0.33, 0))
    bleu_nltk_4 = corpus_bleu(cs_references_list, cs_hypothesis_list, weights=(0.25, 0.25, 0.25, 0.25))


    
    t_references_list = []
    t_hypothesis_list = []

    for ref in s_s_references_list:
        temp = []
        for r in ref:
            temp.append([word2idx.get(i) for i in r])
        t_references_list.append(temp)
        
    
    for hyp in s_s_hypothesis_list:
        t_hypothesis_list.append(word2idx.get(h) for h in hyp)


    pyMetrics_obj = pymetrics(t_references_list,t_hypothesis_list, idx2word)
    bleu_py = pyMetrics_obj.bleu
    bleu_py_1 = bleu_py[0]
    bleu_py_2 = bleu_py[1]
    bleu_py_3 = bleu_py[2]
    bleu_py_4 = bleu_py[3]

    meteor_py = pyMetrics_obj.meteor
    cider_py = pyMetrics_obj.cider
    rouge_py = pyMetrics_obj.rouge


    

    print(f"BLEU (NLTK-1): {bleu_nltk_1}")
    print(f"BLEU (NLTK-2): {bleu_nltk_2}")
    print(f"BLEU (NLTK-3): {bleu_nltk_3}")
    print(f"BLEU (NLTK-4): {bleu_nltk_4}")
    print(f"Meteor: {meteor}")

    print(f"BLEU (Py-1): {bleu_py_1}")
    print(f"BLEU (Py-2): {bleu_py_2}")
    print(f"BLEU (Py-3): {bleu_py_3}")
    print(f"BLEU (Py-4): {bleu_py_4}")
    
    print(f"Meteor (Py): {meteor_py}")
    print(f"CIDEr (Py): {cider_py}")
    print(f"ROUGE (Py): {rouge_py}")

    print(f"BLEU (Sacre): {bleu_sacre}")

    return bleu_nltk_4

# if main

if __name__ == "__main__":
    references_list = [[['this','is','a','test']], [['this', 'is','another', 'test']]]
    hypothesis_list = [['this', 'is','a','test'], ['this', 'is', 'a','test']]
    #references_list = [[[0,1,2,3]], [[0,1,4,3]]]
    #hypothesis_list = [[0,1,2,3], [0,1,2,3]]
    word_map = {'this': 0, 'is': 1, 'a': 2, 'test': 3, 'another': 4}
    rev_word_map = {0: 'this', 1: 'is', 2: 'a', 3: 'test', 4: 'another'}
    #references_list = [['this is a test using paper', 'this is a english test in blue', 'this is a test']]
    #hypothesis_list = ['this is test']
    make_evaluate(references_list, hypothesis_list, rev_word_map, word_map)
