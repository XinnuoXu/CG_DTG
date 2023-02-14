#/bin/bash

test_type=$1


################
# BART-base
################

percent_list=(0.005 0.01 0.05 0.1)
test_from_list=(500 1000 5000 5000)
for i in "${!percent_list[@]}"
do
	percent=${percent_list[i]}
        test_from=${test_from_list[i]}
	base_path=/rds/user/hpcxu1/hpc-work/outputs.webnlg/webnlg_percent_${percent}/
	if [ "$1" = "seen" ]; then
		path_prefix=${base_path}/logs.re.encdec_partial.bbase/test.res.${test_from}
	else
		path_prefix=${base_path}/logs.re.encdec_partial.bbase/test_unseen.res.${test_from}
	fi

	echo '['${test_type}' Baseline]; percent='${percent}
	python ./tool/evaluate_d2t_bleu.py ${path_prefix}
	./tool/multi-bleu.perl ${path_prefix}.bleu_ref1 ${path_prefix}.bleu_ref2 ${path_prefix}.bleu_ref3 < ${path_prefix}.bleu_cand

	python ./tool/parent_data.py ${path_prefix}
	python -m tool.PARENT.table_text_eval --references ${path_prefix}.parent_ref --generations ${path_prefix}.parent_pred --tables ${path_prefix}.parent_table --lambda_weight 0.5

done
echo ''


################
# BART-large
################

percent_list=(0.005 0.01 0.05 0.1)
test_from_list=(500 500 2500 4000)
for i in "${!percent_list[@]}"
do
	percent=${percent_list[i]}
        test_from=${test_from_list[i]}
	base_path=/rds/user/hpcxu1/hpc-work/outputs.webnlg/webnlg_percent_${percent}/
	if [ "$1" = "seen" ]; then
		path_prefix=${base_path}/logs.re.encdec_partial.blarge/test.res.${test_from}
	else
		path_prefix=${base_path}/logs.re.encdec_partial.blarge/test_unseen.res.${test_from}
	fi

	echo '['${test_type}' Baseline]; percent='${percent}
	python ./tool/evaluate_d2t_bleu.py ${path_prefix}
	./tool/multi-bleu.perl ${path_prefix}.bleu_ref1 ${path_prefix}.bleu_ref2 ${path_prefix}.bleu_ref3 < ${path_prefix}.bleu_cand

	python ./tool/parent_data.py ${path_prefix}
	python -m tool.PARENT.table_text_eval --references ${path_prefix}.parent_ref --generations ${path_prefix}.parent_pred --tables ${path_prefix}.parent_table --lambda_weight 0.5

done
echo ''

