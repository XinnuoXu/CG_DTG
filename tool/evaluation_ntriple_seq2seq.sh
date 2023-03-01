#/bin/bash

tokenizer=$1 # [t5-small, t5-base, t5-large]
test_type=$2 # seen unseen

################
# Baseline
################

ntriple_list=(2 3 4 7)
test_from_list=(3000 4000 6000 7000)
#test_from_list=(5000 5000 5000 10000)
#test_from_list=(5000 5000 10000 6000)
for i in "${!ntriple_list[@]}"
do
	ntriple=${ntriple_list[i]}
        test_from=${test_from_list[i]}
	base_path=/rds/user/hpcxu1/hpc-work/outputs.webnlg/${ntriple}triple.full/
	if [ "$test_type" = "seen" ]; then
		path_prefix=${base_path}/logs.re.encdec_partial.${tokenizer}/test.res.${test_from}
	else
		path_prefix=${base_path}/logs.re.encdec_partial.${tokenizer}/test_unseen.res.${test_from}
	fi

	echo '['${test_type}' Baseline]; ntriple='${ntriple}
	python ./tool/evaluate_d2t_bleu.py ${path_prefix}
	./tool/multi-bleu.perl ${path_prefix}.bleu_ref1 ${path_prefix}.bleu_ref2 ${path_prefix}.bleu_ref3 < ${path_prefix}.bleu_cand

	#python ./tool/parent_data.py ${path_prefix}
	#python -m tool.PARENT.table_text_eval --references ${path_prefix}.parent_ref --generations ${path_prefix}.parent_pred --tables ${path_prefix}.parent_table --lambda_weight 0.5

	#python ./tool/nli_preprocess.py ${path_prefix}
	#python ./tool/nli_eval.py --type webnlg ${path_prefix}.nli ./tool/output.json
	#python tool/nli_postprocess.py
done
echo ''

