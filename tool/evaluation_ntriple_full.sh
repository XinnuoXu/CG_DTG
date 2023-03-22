#/bin/bash

test_type=$1
tokenizer='t5-base'


################
# Baseline
################

ntriple_list=(2 3 4 7)
test_from_list=(10000 10000 15000 30000)
for i in "${!ntriple_list[@]}"
do
	ntriple=${ntriple_list[i]}
        test_from=${test_from_list[i]}
	base_path=/rds/user/hpcxu1/hpc-work/outputs.webnlg/${ntriple}triple.full/
	if [ "$1" = "seen" ]; then
		path_prefix=${base_path}/logs.re.encdec_partial.${tokenizer}/test.res.${test_from}
	else
		path_prefix=${base_path}/logs.re.encdec_partial.${tokenizer}/test_unseen.res.${test_from}
	fi

	echo '['${test_type}' Baseline]; ntriple='${ntriple}
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
# Random
################

ntriple_list=(2 3 4 7)
test_from_list=(10000 10000 15000 30000)
for i in "${!ntriple_list[@]}"
do
	ntriple=${ntriple_list[i]}
        test_from=${test_from_list[i]}
	base_path=/rds/user/hpcxu1/hpc-work/outputs.webnlg/${ntriple}triple.full/
	if [ "$1" = "seen" ]; then
		path_prefix=${base_path}/logs.re.random.${tokenizer}/test.res.${test_from}
	else
		path_prefix=${base_path}/logs.re.random.${tokenizer}/test_unseen.res.${test_from}
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

ntriple_list=(2 3 4 7)
test_from_list=(10000 10000 15000 30000)

for i in "${!ntriple_list[@]}"
do
	ntriple=${ntriple_list[i]}
        test_from=${test_from_list[i]}
	base_path=/rds/user/hpcxu1/hpc-work/outputs.webnlg/${ntriple}triple.full/
	if [ "$1" = "seen" ]; then
		path_prefix=${base_path}/logs.re.discriministic.${tokenizer}/test.res.${test_from}
	else
		path_prefix=${base_path}/logs.re.discriministic.${tokenizer}/test_unseen.res.${test_from}
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

echo 'FFN'
ntriple_list=(2 3 4 7)
test_from_list=(500 1500 2000 5500)
for i in "${!ntriple_list[@]}"
do
	ntriple=${ntriple_list[i]}
        test_from=${test_from_list[i]}
	base_path=/rds/user/hpcxu1/hpc-work/outputs.webnlg/${ntriple}triple.full/
	if [ "$1" = "seen" ]; then
		path_prefix=${base_path}/logs.re.nn.ffn_nosample.${tokenizer}/test.res.${test_from}
	else
		path_prefix=${base_path}/logs.re.nn.ffn_nosample.${tokenizer}/test_unseen.res.${test_from}
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

'''

################
# FFN Reinforce
################

echo 'Random Baseline'
ntriple_list=(2 3 4 7)
test_from_list=(4000 6000 3000 6000)
for i in "${!ntriple_list[@]}"
do
	ntriple=${ntriple_list[i]}
        test_from=${test_from_list[i]}
	base_path=/rds/user/hpcxu1/hpc-work/outputs.webnlg/${ntriple}triple.full/
	if [ "$1" = "seen" ]; then
		path_prefix=${base_path}/logs.re.nn.randombase_nosample.${tokenizer}/test.res.${test_from}
	else
		path_prefix=${base_path}/logs.re.nn.randombase_nosample.${tokenizer}/test_unseen.res.${test_from}
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
# FFN Reinforce
################

echo 'Strongbase Baseline'
ntriple_list=(2 3 4 7)
test_from_list=(6000 3000 2000 2000)
for i in "${!ntriple_list[@]}"
do
	ntriple=${ntriple_list[i]}
        test_from=${test_from_list[i]}
	base_path=/rds/user/hpcxu1/hpc-work/outputs.webnlg/${ntriple}triple.full/
	if [ "$1" = "seen" ]; then
		path_prefix=${base_path}/logs.re.nn.strongbase_nosample.${tokenizer}/test.res.${test_from}
	else
		path_prefix=${base_path}/logs.re.nn.strongbase_nosample.${tokenizer}/test_unseen.res.${test_from}
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
# FFN Reinforce
################

echo 'Threshold Baseline'
ntriple_list=(2 3 4 7)
test_from_list=(7000 8000 7000 6000)
for i in "${!ntriple_list[@]}"
do
	ntriple=${ntriple_list[i]}
        test_from=${test_from_list[i]}
	base_path=/rds/user/hpcxu1/hpc-work/outputs.webnlg/${ntriple}triple.full/
	if [ "$1" = "seen" ]; then
		path_prefix=${base_path}/logs.re.nn.thresholdbase_nosample.${tokenizer}/test.res.${test_from}
	else
		path_prefix=${base_path}/logs.re.nn.thresholdbase_nosample.${tokenizer}/test_unseen.res.${test_from}
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

'''
