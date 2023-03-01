#/bin/bash

test_type=$1


################
# Baseline
################

ntriple_list=(2 3 4 5 6 7)
test_from_list=(9000 10000 10000 10000 10000 9000)
for i in "${!ntriple_list[@]}"
do
	ntriple=${ntriple_list[i]}
        test_from=${test_from_list[i]}
	base_path=/rds/user/hpcxu1/hpc-work/outputs.webnlg/${ntriple}triple.single/
	if [ "$1" = "seen" ]; then
		path_prefix=${base_path}/short_large.logs.re.encdec_partial/test.res.${test_from}
	else
		path_prefix=${base_path}/short_large.logs.re.encdec_partial/test_unseen.res.${test_from}
	fi

	echo '['${test_type}' Baseline]; ntriple='${ntriple}
	python ./tool/evaluate_d2t_bleu.py ${path_prefix}
	./tool/multi-bleu.perl ${path_prefix}.bleu_ref1 ${path_prefix}.bleu_ref2 ${path_prefix}.bleu_ref3 < ${path_prefix}.bleu_cand

	python ./tool/parent_data.py ${path_prefix}
	python -m tool.PARENT.table_text_eval --references ${path_prefix}.parent_ref --generations ${path_prefix}.parent_pred --tables ${path_prefix}.parent_table --lambda_weight 0.5

	#python ./tool/nli_preprocess.py ${path_prefix}
	#python ./tool/nli_eval.py --type webnlg ${path_prefix}.nli ./tool/output.json
	#python tool/nli_postprocess.py
done
echo ''

'''
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
		path_prefix=${base_path}/short_large.logs.re.random/test.res.${test_from}
	else
		path_prefix=${base_path}/short_large.logs.re.random/test_unseen.res.${test_from}
	fi

	echo '['${test_type}' Random]; ntriple='${ntriple}
	python ./tool/evaluate_d2t_bleu.py ${path_prefix}
	./tool/multi-bleu.perl ${path_prefix}.bleu_ref1 ${path_prefix}.bleu_ref2 ${path_prefix}.bleu_ref3 < ${path_prefix}.bleu_cand

	python ./tool/parent_data.py ${path_prefix}
	python -m tool.PARENT.table_text_eval --references ${path_prefix}.parent_ref --generations ${path_prefix}.parent_pred --tables ${path_prefix}.parent_table --lambda_weight 0.5

	python ./tool/nli_preprocess.py ${path_prefix}
	python ./tool/nli_eval.py --type webnlg ${path_prefix}.nli ./tool/output.json
	python tool/nli_postprocess.py
done
echo ''

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
		path_prefix=${base_path}/short_large.logs.re.discriministic/test.res.${test_from}
	else
		path_prefix=${base_path}/short_large.logs.re.discriministic/test_unseen.res.${test_from}
	fi

	echo '['${test_type}' Numerical]; ntriple='${ntriple}
	python ./tool/evaluate_d2t_bleu.py ${path_prefix}
	./tool/multi-bleu.perl ${path_prefix}.bleu_ref1 ${path_prefix}.bleu_ref2 ${path_prefix}.bleu_ref3 < ${path_prefix}.bleu_cand

	python ./tool/parent_data.py ${path_prefix}
	python -m tool.PARENT.table_text_eval --references ${path_prefix}.parent_ref --generations ${path_prefix}.parent_pred --tables ${path_prefix}.parent_table --lambda_weight 0.5

	python ./tool/nli_preprocess.py ${path_prefix}
	python ./tool/nli_eval.py --type webnlg ${path_prefix}.nli ./tool/output.json
	python tool/nli_postprocess.py
done
echo ''

################
# FFN
################

ntriple_list=(2 3 4 5 6 7)
test_from_list=(2000 2000 2000 3000 2000 3000)
for i in "${!ntriple_list[@]}"
do
	ntriple=${ntriple_list[i]}
        test_from=${test_from_list[i]}
	base_path=/rds/user/hpcxu1/hpc-work/outputs.webnlg/${ntriple}triple.single/
	if [ "$1" = "seen" ]; then
		path_prefix=${base_path}/short_large.logs.re.nn/test.res.${test_from}
	else
		path_prefix=${base_path}/short_large.logs.re.nn/test_unseen.res.${test_from}
	fi

	echo '['${test_type}' FFN]; ntriple='${ntriple}
	python ./tool/evaluate_d2t_bleu.py ${path_prefix}
	./tool/multi-bleu.perl ${path_prefix}.bleu_ref1 ${path_prefix}.bleu_ref2 ${path_prefix}.bleu_ref3 < ${path_prefix}.bleu_cand

	python ./tool/parent_data.py ${path_prefix}
	python -m tool.PARENT.table_text_eval --references ${path_prefix}.parent_ref --generations ${path_prefix}.parent_pred --tables ${path_prefix}.parent_table --lambda_weight 0.5

	python ./tool/nli_preprocess.py ${path_prefix}
	python ./tool/nli_eval.py --type webnlg ${path_prefix}.nli ./tool/output.json
	python tool/nli_postprocess.py
done
echo ''

################
# FFN Reinforce
################

ntriple_list=(2 3 4 5 6 7)
test_from_list=(1000 1000 4500 3000 1000 3000)
for i in "${!ntriple_list[@]}"
do
	ntriple=${ntriple_list[i]}
        test_from=${test_from_list[i]}
	base_path=/rds/user/hpcxu1/hpc-work/outputs.webnlg/${ntriple}triple.single/
	if [ "$1" = "seen" ]; then
		path_prefix=${base_path}/short_large.logs.re.nn.spectral/test.res.${test_from}
	else
		path_prefix=${base_path}/short_large.logs.re.nn.spectral/test_unseen.res.${test_from}
	fi

	echo '['${test_type}' FFN Reinforce]; ntriple='${ntriple}
	python ./tool/evaluate_d2t_bleu.py ${path_prefix}
	./tool/multi-bleu.perl ${path_prefix}.bleu_ref1 ${path_prefix}.bleu_ref2 ${path_prefix}.bleu_ref3 < ${path_prefix}.bleu_cand

	python ./tool/parent_data.py ${path_prefix}
	python -m tool.PARENT.table_text_eval --references ${path_prefix}.parent_ref --generations ${path_prefix}.parent_pred --tables ${path_prefix}.parent_table --lambda_weight 0.5

	python ./tool/nli_preprocess.py ${path_prefix}
	python ./tool/nli_eval.py --type webnlg ${path_prefix}.nli ./tool/output.json
	python tool/nli_postprocess.py
done
echo ''

################
# FFN Reinforce Sample
################

ntriple_list=(2 3 4 5 6 7)
test_from_list=(500 500 5000 1500 5000 4000)
for i in "${!ntriple_list[@]}"
do
	ntriple=${ntriple_list[i]}
        test_from=${test_from_list[i]}
	base_path=/rds/user/hpcxu1/hpc-work/outputs.webnlg/${ntriple}triple.single/
	if [ "$1" = "seen" ]; then
		path_prefix=${base_path}/short_large.logs.re.nn.spectral_with_sample/test.res.${test_from}
	else
		path_prefix=${base_path}/short_large.logs.re.nn.spectral_with_sample/test_unseen.res.${test_from}
	fi

	echo '['${test_type}' FFN Reinforce sample]; ntriple='${ntriple}
	python ./tool/evaluate_d2t_bleu.py ${path_prefix}
	./tool/multi-bleu.perl ${path_prefix}.bleu_ref1 ${path_prefix}.bleu_ref2 ${path_prefix}.bleu_ref3 < ${path_prefix}.bleu_cand

	python ./tool/parent_data.py ${path_prefix}
	python -m tool.PARENT.table_text_eval --references ${path_prefix}.parent_ref --generations ${path_prefix}.parent_pred --tables ${path_prefix}.parent_table --lambda_weight 0.5

	python ./tool/nli_preprocess.py ${path_prefix}
	python ./tool/nli_eval.py --type webnlg ${path_prefix}.nli ./tool/output.json
	python tool/nli_postprocess.py
done
'''
