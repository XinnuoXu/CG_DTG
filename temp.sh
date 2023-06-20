
echo 'Random Baseline'
#ntriple_list=(2 3 4 7)
#test_from_list=(2000 7000 5000 4000)
mtriple_list=( 7)
test_from_list=( 4000)
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

	if [ "$copy_data" = true ]; then
                target_dir=${saved_path}/${ntriple}triple.full/rl/
                source_output=${path_prefix}.*
                source_model=${base_path}/model.re.nn.spectral.${tokenizer}/model_step_${test_from}.pt
                mkdir -p ${target_dir}
                cp ${source_output} ${target_dir}
                cp ${source_model} ${target_dir}
        fi
done
echo ''


