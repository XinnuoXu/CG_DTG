#/bin/bash

test_type=$1

################
# Baseline
################

ntriple_list=(2 3 4 5 6 7)
test_from_list=(1000 2000 2000 3000 3000 3000)
for i in "${!ntriple_list[@]}"
do
	ntriple=${ntriple_list[i]}
        test_from=${test_from_list[i]}
	base_path=/rds/user/hpcxu1/hpc-work/outputs.webnlg/${ntriple}triple.single/
	if [ "$1" = "seen" ]; then
		path_prefix=${base_path}/short_single.logs.re.encdec_partial/test.res.${test_from}
	else
		path_prefix=${base_path}/short_single.logs.re.encdec_partial/test_unseen.res.${test_from}
	fi

	echo 'BLEU for ['${test_type}' Baseline]; ntriple='${ntriple}
	python ./tool/evaluate_d2t_bleu.py ${path_prefix}
	./tool/multi-bleu.perl ${path_prefix}.bleu_ref1 ${path_prefix}.bleu_ref2 ${path_prefix}.bleu_ref3 < ${path_prefix}.bleu_cand

	echo 'PARENT for ['${test_type}' Baseline]; ntriple='${ntriple}
	python ./tool/parent_data.py ${path_prefix}
	python -m tool.PARENT.table_text_eval --references ${path_prefix}.parent_ref --generations ${path_prefix}.parent_pred --tables ${path_prefix}.parent_table --lambda_weight 0.5
done

################
# Random
################

ntriple_list=(2 3 4 5 6 7)
test_from_list=(1000 2000 2000 3000 3000 3000)
for i in "${!ntriple_list[@]}"
do
	ntriple=${ntriple_list[i]}
        test_from=${test_from_list[i]}
	base_path=/rds/user/hpcxu1/hpc-work/outputs.webnlg/${ntriple}triple.single/
	if [ "$1" = "seen" ]; then
		path_prefix=${base_path}/short_single.logs.re.random/test.res.${test_from}
	else
		path_prefix=${base_path}/short_single.logs.re.random/test_unseen.res.${test_from}
	fi

	echo 'BLEU for ['${test_type}' Random]; ntriple='${ntriple}
	python ./tool/evaluate_d2t_bleu.py ${path_prefix}
	./tool/multi-bleu.perl ${path_prefix}.bleu_ref1 ${path_prefix}.bleu_ref2 ${path_prefix}.bleu_ref3 < ${path_prefix}.bleu_cand


	echo 'PARENT for ['${test_type}' Random]; ntriple='${ntriple}
	python ./tool/parent_data.py ${path_prefix}
	python -m tool.PARENT.table_text_eval --references ${path_prefix}.parent_ref --generations ${path_prefix}.parent_pred --tables ${path_prefix}.parent_table --lambda_weight 0.5
done

################
# Numerical
################

ntriple_list=(2 3 4 5 6 7)
test_from_list=(1000 2000 2000 3000 3000 3000)
for i in "${!ntriple_list[@]}"
do
	ntriple=${ntriple_list[i]}
        test_from=${test_from_list[i]}
	base_path=/rds/user/hpcxu1/hpc-work/outputs.webnlg/${ntriple}triple.single/
	if [ "$1" = "seen" ]; then
		path_prefix=${base_path}/short_single.logs.re.discriministic/test.res.${test_from}
	else
		path_prefix=${base_path}/short_single.logs.re.discriministic/test_unseen.res.${test_from}
	fi

	echo 'BLEU for ['${test_type}' Numerical]; ntriple='${ntriple}
	python ./tool/evaluate_d2t_bleu.py ${path_prefix}
	./tool/multi-bleu.perl ${path_prefix}.bleu_ref1 ${path_prefix}.bleu_ref2 ${path_prefix}.bleu_ref3 < ${path_prefix}.bleu_cand

	echo 'PARENT for ['${test_type}' Numerical]; ntriple='${ntriple}
	python ./tool/parent_data.py ${path_prefix}
	python -m tool.PARENT.table_text_eval --references ${path_prefix}.parent_ref --generations ${path_prefix}.parent_pred --tables ${path_prefix}.parent_table --lambda_weight 0.5
done

################
# FFN
################

ntriple_list=(2 3 4 5 6 7)
test_from_list=(2000 2000 3000 4000 2000 3000)
for i in "${!ntriple_list[@]}"
do
	ntriple=${ntriple_list[i]}
        test_from=${test_from_list[i]}
	base_path=/rds/user/hpcxu1/hpc-work/outputs.webnlg/${ntriple}triple.single/
	if [ "$1" = "seen" ]; then
		path_prefix=${base_path}/short_single.logs.re.nn/test.res.${test_from}
	else
		path_prefix=${base_path}/short_single.logs.re.nn/test_unseen.res.${test_from}
	fi

	echo 'BLEU for ['${test_type}' FFN]; ntriple='${ntriple}
	python ./tool/evaluate_d2t_bleu.py ${path_prefix}
	./tool/multi-bleu.perl ${path_prefix}.bleu_ref1 ${path_prefix}.bleu_ref2 ${path_prefix}.bleu_ref3 < ${path_prefix}.bleu_cand

	echo 'PARENT for ['${test_type}' FFN]; ntriple='${ntriple}
	python ./tool/parent_data.py ${path_prefix}
	python -m tool.PARENT.table_text_eval --references ${path_prefix}.parent_ref --generations ${path_prefix}.parent_pred --tables ${path_prefix}.parent_table --lambda_weight 0.5
done

################
# FFN Reinforce
################

ntriple_list=(2 3 4 5 6 7)
test_from_list=(3500 5000 4000 4500 2500 4000)
for i in "${!ntriple_list[@]}"
do
	ntriple=${ntriple_list[i]}
        test_from=${test_from_list[i]}
	base_path=/rds/user/hpcxu1/hpc-work/outputs.webnlg/${ntriple}triple.single/
	if [ "$1" = "seen" ]; then
		path_prefix=${base_path}/short_single.logs.re.nn.spectral/test.res.${test_from}
	else
		path_prefix=${base_path}/short_single.logs.re.nn.spectral/test_unseen.res.${test_from}
	fi

	echo 'BLEU for ['${test_type}' FFN Reinforce]; ntriple='${ntriple}
	python ./tool/evaluate_d2t_bleu.py ${path_prefix}
	./tool/multi-bleu.perl ${path_prefix}.bleu_ref1 ${path_prefix}.bleu_ref2 ${path_prefix}.bleu_ref3 < ${path_prefix}.bleu_cand

	echo 'PARENT for ['${test_type}' FFN Reinforce]; ntriple='${ntriple}
	python ./tool/parent_data.py ${path_prefix}
	python -m tool.PARENT.table_text_eval --references ${path_prefix}.parent_ref --generations ${path_prefix}.parent_pred --tables ${path_prefix}.parent_table --lambda_weight 0.5
done

################
# FFN Reinforce Sample
################

ntriple_list=(2 3 4 5 6 7)
test_from_list=(3500 4500 3500 4500 2500 3500)
for i in "${!ntriple_list[@]}"
do
	ntriple=${ntriple_list[i]}
        test_from=${test_from_list[i]}
	base_path=/rds/user/hpcxu1/hpc-work/outputs.webnlg/${ntriple}triple.single/
	if [ "$1" = "seen" ]; then
		path_prefix=${base_path}/short_single.logs.re.nn.spectral_with_sample/test.res.${test_from}
	else
		path_prefix=${base_path}/short_single.logs.re.nn.spectral_with_sample/test_unseen.res.${test_from}
	fi

	echo 'BLEU for ['${test_type}' FFN Reinforce sample]; ntriple='${ntriple}
	python ./tool/evaluate_d2t_bleu.py ${path_prefix}
	./tool/multi-bleu.perl ${path_prefix}.bleu_ref1 ${path_prefix}.bleu_ref2 ${path_prefix}.bleu_ref3 < ${path_prefix}.bleu_cand

	echo 'PARENT for ['${test_type}' FFN Reinforce sample]; ntriple='${ntriple}
	python ./tool/parent_data.py ${path_prefix}
	python -m tool.PARENT.table_text_eval --references ${path_prefix}.parent_ref --generations ${path_prefix}.parent_pred --tables ${path_prefix}.parent_table --lambda_weight 0.5
done
