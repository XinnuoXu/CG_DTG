#coding=utf8

import json
import matplotlib.pyplot as plt
import numpy as np
import random

if __name__ == '__main__':
    entropy_mapping = {}
    numbers = [json.loads(line.strip()) for line in open('entropy_analysis.json')]
    for obj in numbers:
        for key in obj:
            if key not in entropy_mapping:
                entropy_mapping[key] = []
            entropy_mapping[key].append(obj[key])
    min_size = min([len(entropy_mapping[key]) for key in entropy_mapping])
    for key in entropy_mapping:
        sampled_idx = random.sample([i for i in range(len(entropy_mapping[key]))], min_size-1)
        sampled_idx = [0] + sorted(sampled_idx)
        entropies = [entropy_mapping[key][idx] for idx in sampled_idx]
        ypoints = np.array(entropies)
        plt.plot(ypoints, label = f'nsent_{key}')
    plt.ylabel('Entropy')
    plt.legend()
    plt.savefig('entropy_analysis.png')

