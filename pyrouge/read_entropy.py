#coding=utf8

import json
import matplotlib.pyplot as plt
import numpy as np
import random

def Plot_one_file(filename, linestyles, tag):
    entropy_mapping = {}
    numbers = []
    for line in open(filename):
        try:
            obj = json.loads(line.strip())
        except:
            continue
        numbers.append(obj)

    for obj in numbers:
        for key in obj:
            if key not in entropy_mapping:
                entropy_mapping[key] = []
            entropy_mapping[key].append(obj[key])
    min_size = min([len(entropy_mapping[key]) for key in entropy_mapping if len(entropy_mapping[key])>5])
    for key in entropy_mapping:
        if len(entropy_mapping[key]) < min_size:
            continue
        sampled_idx = random.sample([i for i in range(len(entropy_mapping[key]))], min_size-1)
        sampled_idx = [0] + sorted(sampled_idx)
        entropies = [entropy_mapping[key][idx] for idx in sampled_idx]
        ypoints = np.array(entropies)
        plt.plot(ypoints, label = f'{tag}_nsent_{key}', linestyle=linestyles)


if __name__ == '__main__':
    #Plot_one_file('entropy_analysis.json', 'solid', 'softmax')
    Plot_one_file('entropy_analysis.marginal', 'dotted', 'marginal')
    #Plot_one_file('entropy_analysis.marginal.kmeans', 'solid', 'soft-kmeans')
    #Plot_one_file('entropy_analysis.marginal.schedule', 'solid', 'marginal_schedule')
    #Plot_one_file('entropy_analysis.plan', 'dotted', 'sparsemax')
    #Plot_one_file('entropy_analysis.sample', 'dotted', 'sparsemax')
    #Plot_one_file('entropy_analysis.sparsemax_fromraw', 'dashed', 'sparsemax_raw')

    plt.ylabel('Entropy')
    plt.legend()
    plt.savefig('entropy_analysis_plan.png')

