import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(parent_dir)

from utils.metrics import make_evaluate

references_list = [[[0,1,2,3]], [[0,1,4,3]]]
hypothesis_list = [[0,1,2,3], [0,1,2,3]]
word_map = {'this': 0, 'is': 1, 'a': 2, 'test': 3, 'another': 4}
rev_word_map = {0: 'this', 1: 'is', 2: 'a', 3: 'test', 4: 'another'}

make_evaluate(references_list, hypothesis_list, rev_word_map)

