#coding=utf8

import sys

if __name__ == '__main__':
	nums = []
	for line in open(sys.argv[1]):
		flist = line.strip().split('\t')
		groups = flist[2].split('|||')
		num_of_groups = len(groups)
		nums.append(num_of_groups)
	print ('#Avg number of groups:', sum(nums)/float(len(nums)))
