#coding=utf8 

#file_list = ['777.random', '777.ffn']
file_list = ['111.random', '111.ffn']

if __name__ == '__main__':
	import matplotlib.pyplot as plt
	import numpy as np
	import sys

	for filname in file_list:
		x = [float(line.strip()) for line in open(filname)]
		plt.hist(x, bins=np.arange(0, 1, 0.05), alpha=0.5, label=filname)
	plt.legend()
	#plt.savefig('hist777.pdf')
	plt.savefig('hist111.pdf')
