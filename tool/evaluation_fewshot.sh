#/bin/bash

test_type="seen"
copy_data=false
saved_path=/rds/user/hpcxu1/hpc-work/outputs.webnlg/saved_data/

'''
################
# BART-base
################

percent_list=(0.005 0.01 0.05 0.1)
test_from_list=(500 500 2500 4500)
for i in "${!percent_list[@]}"
do
	percent=${percent_list[i]}
        test_from=${test_from_list[i]}
	base_path=/rds/user/hpcxu1/hpc-work/outputs.webnlg/webnlg_percent_${percent}/
	if [ "$test_type" = "seen" ]; then
		path_prefix=${base_path}/logs.re.encdec_partial/test.res.${test_from}
	else
		path_prefix=${base_path}/logs.re.encdec_partial/test_unseen.res.${test_from}
	fi

	echo '['${test_type}' Baseline]; percent='${percent}
	python ./tool/evaluate_d2t_bleu.py ${path_prefix}
	./tool/multi-bleu.perl ${path_prefix}.bleu_ref1 ${path_prefix}.bleu_ref2 ${path_prefix}.bleu_ref3 < ${path_prefix}.bleu_cand

	python ./tool/parent_data.py ${path_prefix}
	python -m tool.PARENT.table_text_eval --references ${path_prefix}.parent_ref --generations ${path_prefix}.parent_pred --tables ${path_prefix}.parent_table --lambda_weight 0.5

	#python ./tool/nli_preprocess.py ${path_prefix}
	#python ./tool/nli_eval.py --type webnlg ${path_prefix}.nli ./tool/output.json
	#python tool/nli_postprocess.py

	if [ "$copy_data" = true ]; then
		target_dir=${saved_path}/webnlg_percent_${percent}/encdec_partial/
		source_output=${base_path}/logs.re.encdec_partial/test.res.${test_from}.*
		source_model=${base_path}/model.re.encdec_partial/model_step_${test_from}.pt
		mkdir -p ${target_dir}
		cp ${source_output} ${target_dir}
		cp ${source_model} ${target_dir}
	fi
		
done
echo ''


################
# Random
################

percent_list=(0.005 0.01 0.05 0.1)
test_from_list=(500 500 2500 4500)
for i in "${!percent_list[@]}"
do
	percent=${percent_list[i]}
        test_from=${test_from_list[i]}
	base_path=/rds/user/hpcxu1/hpc-work/outputs.webnlg/webnlg_percent_${percent}/
	if [ "$test_type" = "seen" ]; then
		path_prefix=${base_path}/logs.re.random/test.res.${test_from}
	else
		path_prefix=${base_path}/logs.re.random/test_unseen.res.${test_from}
	fi

	echo '['${test_type}' Random]; percent='${percent}
	python ./tool/evaluate_d2t_bleu.py ${path_prefix}
	./tool/multi-bleu.perl ${path_prefix}.bleu_ref1 ${path_prefix}.bleu_ref2 ${path_prefix}.bleu_ref3 < ${path_prefix}.bleu_cand

	python ./tool/parent_data.py ${path_prefix}
	python -m tool.PARENT.table_text_eval --references ${path_prefix}.parent_ref --generations ${path_prefix}.parent_pred --tables ${path_prefix}.parent_table --lambda_weight 0.5

	#python ./tool/nli_preprocess.py ${path_prefix}
	#python ./tool/nli_eval.py --type webnlg ${path_prefix}.nli ./tool/output.json
	#python tool/nli_postprocess.py

	if [ "$copy_data" = true ]; then
		target_dir=${saved_path}/webnlg_percent_${percent}/random/
		source_output=${base_path}/logs.re.random/test.res.${test_from}.*
		source_model=${base_path}/model.re.encdec_partial/model_step_${test_from}.pt
		mkdir -p ${target_dir}
		cp ${source_output} ${target_dir}
		cp ${source_model} ${target_dir}
	fi
done
echo ''


################
# Numerical
################

percent_list=(0.005 0.01 0.05 0.1)
test_from_list=(500 500 2500 4500)
for i in "${!percent_list[@]}"
do
	percent=${percent_list[i]}
        test_from=${test_from_list[i]}
	base_path=/rds/user/hpcxu1/hpc-work/outputs.webnlg/webnlg_percent_${percent}/
	if [ "$test_type" = "seen" ]; then
		path_prefix=${base_path}/logs.re.discriministic/test.res.${test_from}
	else
		path_prefix=${base_path}/logs.re.discriministic/test_unseen.res.${test_from}
	fi

	echo '['${test_type}' Numerical]; percent='${percent}
	python ./tool/evaluate_d2t_bleu.py ${path_prefix}
	./tool/multi-bleu.perl ${path_prefix}.bleu_ref1 ${path_prefix}.bleu_ref2 ${path_prefix}.bleu_ref3 < ${path_prefix}.bleu_cand

	python ./tool/parent_data.py ${path_prefix}
	python -m tool.PARENT.table_text_eval --references ${path_prefix}.parent_ref --generations ${path_prefix}.parent_pred --tables ${path_prefix}.parent_table --lambda_weight 0.5

	#python ./tool/nli_preprocess.py ${path_prefix}
	#python ./tool/nli_eval.py --type webnlg ${path_prefix}.nli ./tool/output.json
	#python tool/nli_postprocess.py

	if [ "$copy_data" = true ]; then
		target_dir=${saved_path}/webnlg_percent_${percent}/discriministic/
		source_output=${base_path}/logs.re.discriministic/test.res.${test_from}.*
		source_model=${base_path}/model.re.encdec_partial/model_step_${test_from}.pt
		mkdir -p ${target_dir}
		cp ${source_output} ${target_dir}
		cp ${source_model} ${target_dir}
	fi
done
echo ''
'''


################
# FNN
################

percent_list=(0.005 0.01 0.05 0.1)
test_from_list=(500 500 500 1000)
for i in "${!percent_list[@]}"
do
	percent=${percent_list[i]}
        test_from=${test_from_list[i]}
	base_path=/rds/user/hpcxu1/hpc-work/outputs.webnlg/webnlg_percent_${percent}/
	if [ "$test_type" = "seen" ]; then
		path_prefix=${base_path}/logs.re.nn/test.res.${test_from}
	else
		path_prefix=${base_path}/logs.re.nn/test_unseen.res.${test_from}
	fi

	echo '['${test_type}' FNN]; percent='${percent}
	python ./tool/evaluate_d2t_bleu.py ${path_prefix}
	./tool/multi-bleu.perl ${path_prefix}.bleu_ref1 ${path_prefix}.bleu_ref2 ${path_prefix}.bleu_ref3 < ${path_prefix}.bleu_cand

	python ./tool/parent_data.py ${path_prefix}
	python -m tool.PARENT.table_text_eval --references ${path_prefix}.parent_ref --generations ${path_prefix}.parent_pred --tables ${path_prefix}.parent_table --lambda_weight 0.5

	python ./tool/nli_preprocess.py ${path_prefix}
	python ./tool/nli_eval.py --type webnlg ${path_prefix}.nli ./tool/output.json
	python tool/nli_postprocess.py

	if [ "$copy_data" = true ]; then
		target_dir=${saved_path}/webnlg_percent_${percent}/nn/
		source_output=${base_path}/logs.re.nn/test.res.${test_from}.*
		source_model=${base_path}/model.re.nn/model_step_${test_from}.pt
		mkdir -p ${target_dir}
		cp ${source_output} ${target_dir}
		cp ${source_model} ${target_dir}
	fi
done
echo ''



'''
################
# FNN
################

percent_list=(0.005 0.01 0.05 0.1)
test_from_list=(1500 1000 2500 1000)
for i in "${!percent_list[@]}"
do
	percent=${percent_list[i]}
        test_from=${test_from_list[i]}
	base_path=/rds/user/hpcxu1/hpc-work/outputs.webnlg/webnlg_percent_${percent}/
	if [ "$test_type" = "seen" ]; then
		path_prefix=${base_path}/logs.re.nn.spectral//test.res.${test_from}
	else
		path_prefix=${base_path}/logs.re.nn.spectral/test_unseen.res.${test_from}
	fi

	echo '['${test_type}' RL]; percent='${percent}
	python ./tool/evaluate_d2t_bleu.py ${path_prefix}
	./tool/multi-bleu.perl ${path_prefix}.bleu_ref1 ${path_prefix}.bleu_ref2 ${path_prefix}.bleu_ref3 < ${path_prefix}.bleu_cand

	python ./tool/parent_data.py ${path_prefix}
	python -m tool.PARENT.table_text_eval --references ${path_prefix}.parent_ref --generations ${path_prefix}.parent_pred --tables ${path_prefix}.parent_table --lambda_weight 0.5

	python ./tool/nli_preprocess.py ${path_prefix}
	python ./tool/nli_eval.py --type webnlg ${path_prefix}.nli ./tool/output.json
	python tool/nli_postprocess.py

	if [ "$copy_data" = true ]; then
		target_dir=${saved_path}/webnlg_percent_${percent}/nn/
		source_output=${base_path}/logs.re.nn.spectral//test.res.${test_from}.*
		source_model=${base_path}/model.re.nn.spectral/model_step_${test_from}.pt
		mkdir -p ${target_dir}
		cp ${source_output} ${target_dir}
		cp ${source_model} ${target_dir}
	fi
done
echo ''
'''
