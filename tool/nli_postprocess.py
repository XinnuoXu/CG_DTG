#coding=utf8

import json 

if __name__ == '__main__':

	with open('./tool/output.json', 'r', encoding='UTF-8') as fh:
                res = json.load(fh)

	mr2sent_entail_scores = []
	sent2mr_entail_scores = []
	tag_dict = {}
	for example in res:
		result_tag = example['result']
		mr2sent_entail_score = float(example['raw_results']['mr2sent'].split('E: ')[1])
		sent2mr_entail_score = float(example['raw_results']['sent2mr'].split('E: ')[1])
		if result_tag not in tag_dict:
			tag_dict[result_tag] = 0
		tag_dict[result_tag] += 1
		mr2sent_entail_scores.append(mr2sent_entail_score)
		sent2mr_entail_scores.append(sent2mr_entail_score)

	for key in tag_dict:
		tag_dict[key] = tag_dict[key]/len(res)
	mr2sent_entail_score = sum(mr2sent_entail_scores)/len(mr2sent_entail_scores)
	sent2mr_entail_score = sum(sent2mr_entail_scores)/len(sent2mr_entail_scores)
	print ('OK_percent: %.4f; omission: %.4f; hallucination: %.4f; hallucination+omission: %.4f; mr2sent_entail_score: %.4f; sent2mr_entail_score: %.4f' % (tag_dict['OK'], tag_dict['omission'], tag_dict['hallucination'], tag_dict['hallucination+omission'], mr2sent_entail_score, sent2mr_entail_score))
