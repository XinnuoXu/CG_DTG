#coding=utf8

import sys 
from utils import rouge_results_to_str, test_rouge

TEMP_DIR = './temp/'
CAN_PATH = sys.argv[1]
GOLE_PATH = sys.argv[2]

if __name__ == '__main__':
    results_dict = test_rouge(TEMP_DIR, CAN_PATH, GOLE_PATH)
    rouge_results_to_str(results_dict)
