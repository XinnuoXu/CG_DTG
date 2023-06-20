#/bin/bash

test_type=$1
copy_data=false
saved_path=/rds/user/hpcxu1/hpc-work/outputs.webnlg/saved_data/

'''
################
# Baseline
################

ntriple_list=(2 3 4 5 6 7)
test_from_list=(2000 3000 4000 5000 6000 8000)
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

	echo '['${test_type}' Baseline]; ntriple='${ntriple}
	python ./tool/evaluate_d2t_bleu.py ${path_prefix}
	./tool/multi-bleu.perl ${path_prefix}.bleu_ref1 ${path_prefix}.bleu_ref2 ${path_prefix}.bleu_ref3 < ${path_prefix}.bleu_cand

	#python ./tool/parent_data.py ${path_prefix}
	#python -m tool.PARENT.table_text_eval --references ${path_prefix}.parent_ref --generations ${path_prefix}.parent_pred --tables ${path_prefix}.parent_table --lambda_weight 0.5

	#python ./tool/nli_preprocess.py ${path_prefix}
	#python ./tool/nli_eval.py --type webnlg ${path_prefix}.nli ./tool/output.json
	#python tool/nli_postprocess.py

	if [ "$copy_data" = true ]; then
                target_dir=${saved_path}/${ntriple}triple.single/encdec_partial/
                source_output=${path_prefix}.*
                source_model=${base_path}/short_single.model.re.encdec_partial//model_step_${test_from}.pt
                mkdir -p ${target_dir}
		rm ${target_dir}/*
                cp ${source_output} ${target_dir}
                cp ${source_model} ${target_dir}
        fi
done
echo ''
'''

################
# Random
################

ntriple_list=(2 3 4 5 6 7)
test_from_list=(2000 3000 4000 5000 6000 8000)
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

	echo '['${test_type}' Random]; ntriple='${ntriple}
	python ./tool/evaluate_d2t_bleu.py ${path_prefix}
	./tool/multi-bleu.perl ${path_prefix}.bleu_ref1 ${path_prefix}.bleu_ref2 ${path_prefix}.bleu_ref3 < ${path_prefix}.bleu_cand

	#python ./tool/parent_data.py ${path_prefix}
	#python -m tool.PARENT.table_text_eval --references ${path_prefix}.parent_ref --generations ${path_prefix}.parent_pred --tables ${path_prefix}.parent_table --lambda_weight 0.5

	#python ./tool/nli_preprocess.py ${path_prefix}
	#python ./tool/nli_eval.py --type webnlg ${path_prefix}.nli ./tool/output.json
	#python tool/nli_postprocess.py

	python ./tool/evaluate_cluster_num.py ${path_prefix}.cluster

	if [ "$copy_data" = true ]; then
                target_dir=${saved_path}/${ntriple}triple.single/random/
                source_output=${path_prefix}.*
                source_model=${base_path}/short_single.model.re.encdec_partial//model_step_${test_from}.pt
                mkdir -p ${target_dir}
		rm ${target_dir}/*
                cp ${source_output} ${target_dir}
                cp ${source_model} ${target_dir}
        fi
done
echo ''

################
# Numerical
################

ntriple_list=(2 3 4 5 6 7)
test_from_list=(2000 3000 4000 5000 6000 8000)
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

	echo '['${test_type}' Numerical]; ntriple='${ntriple}
	python ./tool/evaluate_d2t_bleu.py ${path_prefix}
	./tool/multi-bleu.perl ${path_prefix}.bleu_ref1 ${path_prefix}.bleu_ref2 ${path_prefix}.bleu_ref3 < ${path_prefix}.bleu_cand

	#python ./tool/parent_data.py ${path_prefix}
	#python -m tool.PARENT.table_text_eval --references ${path_prefix}.parent_ref --generations ${path_prefix}.parent_pred --tables ${path_prefix}.parent_table --lambda_weight 0.5

	#python ./tool/nli_preprocess.py ${path_prefix}
	#python ./tool/nli_eval.py --type webnlg ${path_prefix}.nli ./tool/output.json
	#python tool/nli_postprocess.py

	python ./tool/evaluate_cluster_num.py ${path_prefix}.cluster

	if [ "$copy_data" = true ]; then
                target_dir=${saved_path}/${ntriple}triple.single/discriministic/
                source_output=${path_prefix}.*
                source_model=${base_path}/short_single.model.re.encdec_partial//model_step_${test_from}.pt
                mkdir -p ${target_dir}
		rm ${target_dir}/*
                cp ${source_output} ${target_dir}
                cp ${source_model} ${target_dir}
        fi
done
echo ''


################
# FFN
################

ntriple_list=(2 3 4 5 6 7)
test_from_list=(500 500 500 1000 2000 2000)
for i in "${!ntriple_list[@]}"
do
	ntriple=${ntriple_list[i]}
        test_from=${test_from_list[i]}
	base_path=/rds/user/hpcxu1/hpc-work/outputs.webnlg/${ntriple}triple.single/
	if [ "$1" = "seen" ]; then
		path_prefix=${base_path}/short_single.logs.re.nn.ffn_nosample/test.res.${test_from}
	else
		path_prefix=${base_path}/short_single.logs.re.nn.ffn_nosample/test_unseen.res.${test_from}
	fi

	echo '['${test_type}' FFN]; ntriple='${ntriple}
	python ./tool/evaluate_d2t_bleu.py ${path_prefix}
	./tool/multi-bleu.perl ${path_prefix}.bleu_ref1 ${path_prefix}.bleu_ref2 ${path_prefix}.bleu_ref3 < ${path_prefix}.bleu_cand

	#python ./tool/parent_data.py ${path_prefix}
	#python -m tool.PARENT.table_text_eval --references ${path_prefix}.parent_ref --generations ${path_prefix}.parent_pred --tables ${path_prefix}.parent_table --lambda_weight 0.5

	#python ./tool/nli_preprocess.py ${path_prefix}
	#python ./tool/nli_eval.py --type webnlg ${path_prefix}.nli ./tool/output.json
	#python tool/nli_postprocess.py

	python ./tool/evaluate_cluster_num.py ${path_prefix}.cluster

	if [ "$copy_data" = true ]; then
                target_dir=${saved_path}/${ntriple}triple.single/nn/
                source_output=${path_prefix}.*
                source_model=${base_path}/short_single.model.re.nn/model_step_${test_from}.pt
                mkdir -p ${target_dir}
		rm ${target_dir}/*
                cp ${source_output} ${target_dir}
                cp ${source_model} ${target_dir}
        fi
done
echo ''


################
# FFN Reinforce
################

echo '[Randombase]'
#ntriple_list=(2 3 4 5 6 7)
#test_from_list=(2000 1500 1000 2500 2000 2000)
#ntriple_list=(2 3 4 5)
#test_from_list=(3000 1500 1500 2500)
ntriple_list=(2 3 4 5 6 7)
test_from_list=(500 1500 500 2500 2000 2000)
for i in "${!ntriple_list[@]}"
do
	ntriple=${ntriple_list[i]}
        test_from=${test_from_list[i]}
	base_path=/rds/user/hpcxu1/hpc-work/outputs.webnlg/${ntriple}triple.single/
	if [ "$1" = "seen" ]; then
		path_prefix=${base_path}/short_single.logs.re.nn.randombase_nosample/test.res.${test_from}
	else
		path_prefix=${base_path}/short_single.logs.re.nn.randombase_nosample/test_unseen.res.${test_from}
	fi

	echo '['${test_type}' FFN Reinforce]; ntriple='${ntriple}
	python ./tool/evaluate_d2t_bleu.py ${path_prefix}
	./tool/multi-bleu.perl ${path_prefix}.bleu_ref1 ${path_prefix}.bleu_ref2 ${path_prefix}.bleu_ref3 < ${path_prefix}.bleu_cand

	#python ./tool/parent_data.py ${path_prefix}
	#python -m tool.PARENT.table_text_eval --references ${path_prefix}.parent_ref --generations ${path_prefix}.parent_pred --tables ${path_prefix}.parent_table --lambda_weight 0.5

	#python ./tool/nli_preprocess.py ${path_prefix}
	#python ./tool/nli_eval.py --type webnlg ${path_prefix}.nli ./tool/output.json
	#python tool/nli_postprocess.py

	python ./tool/evaluate_cluster_num.py ${path_prefix}.cluster

	if [ "$copy_data" = true ]; then
                target_dir=${saved_path}/${ntriple}triple.single/rl/
                source_output=${path_prefix}.*
                source_model=${base_path}/short_single.model.re.nn.randombase/model_step_${test_from}.pt
                mkdir -p ${target_dir}
		rm ${target_dir}/*
                cp ${source_output} ${target_dir}
                cp ${source_model} ${target_dir}
        fi
done
echo ''
