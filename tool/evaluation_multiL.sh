#/bin/bash

test_type="seen"


################
# BART-base
################

lang_list=(br cy ga ru)
test_from_list=(5000 5000 8000 7000)
#test_from_list=(10000 10000 8000 7000)
for i in "${!lang_list[@]}"
do
	lang=${lang_list[i]}
        test_from=${test_from_list[i]}
	base_path=/rds/user/hpcxu1/hpc-work/outputs.webnlg/${lang}triple.full/
	if [ "$test_type" = "seen" ]; then
		path_prefix=${base_path}/logs.re.encdec_partial/test.res.${test_from}
	else
		path_prefix=${base_path}/logs.re.encdec_partial/test_unseen.res.${test_from}
	fi

	echo '['${test_type}' Baseline]; lang='${lang}
	python ./tool/evaluate_d2t_bleu.py ${path_prefix}
	./tool/multi-bleu.perl ${path_prefix}.bleu_ref1 ${path_prefix}.bleu_ref2 ${path_prefix}.bleu_ref3 < ${path_prefix}.bleu_cand

	#python ./tool/parent_data.py ${path_prefix}
	#python -m tool.PARENT.table_text_eval --references ${path_prefix}.parent_ref --generations ${path_prefix}.parent_pred --tables ${path_prefix}.parent_table --lambda_weight 0.5

	#python ./tool/nli_preprocess.py ${path_prefix}
	#python ./tool/nli_eval.py --type webnlg ${path_prefix}.nli ./tool/output.json
	#python tool/nli_postprocess.py
done
echo ''

'''
################
# Random
################

lang_list=(br cy ga ru)
test_from_list=(5000 5000 8000 7000)
for i in "${!lang_list[@]}"
do
	lang=${lang_list[i]}
        test_from=${test_from_list[i]}
	base_path=/rds/user/hpcxu1/hpc-work/outputs.webnlg/${lang}triple.full/
	if [ "$test_type" = "seen" ]; then
		path_prefix=${base_path}/logs.re.random/test.res.${test_from}
	else
		path_prefix=${base_path}/logs.re.random/test_unseen.res.${test_from}
	fi

	echo '['${test_type}' Random]; lang='${lang}
	python ./tool/evaluate_d2t_bleu.py ${path_prefix}
	./tool/multi-bleu.perl ${path_prefix}.bleu_ref1 ${path_prefix}.bleu_ref2 ${path_prefix}.bleu_ref3 < ${path_prefix}.bleu_cand

	#python ./tool/parent_data.py ${path_prefix}
	#python -m tool.PARENT.table_text_eval --references ${path_prefix}.parent_ref --generations ${path_prefix}.parent_pred --tables ${path_prefix}.parent_table --lambda_weight 0.5

	#python ./tool/nli_preprocess.py ${path_prefix}
	#python ./tool/nli_eval.py --type webnlg ${path_prefix}.nli ./tool/output.json
	#python tool/nli_postprocess.py
done
echo ''


################
# Numerical
################

lang_list=(br cy ga ru)
test_from_list=(5000 5000 8000 7000)
for i in "${!lang_list[@]}"
do
	lang=${lang_list[i]}
        test_from=${test_from_list[i]}
	base_path=/rds/user/hpcxu1/hpc-work/outputs.webnlg/${lang}triple.full/
	if [ "$test_type" = "seen" ]; then
		path_prefix=${base_path}/logs.re.discriministic/test.res.${test_from}
	else
		path_prefix=${base_path}/logs.re.discriministic/test_unseen.res.${test_from}
	fi

	echo '['${test_type}' Numerical]; lang='${lang}
	python ./tool/evaluate_d2t_bleu.py ${path_prefix}
	./tool/multi-bleu.perl ${path_prefix}.bleu_ref1 ${path_prefix}.bleu_ref2 ${path_prefix}.bleu_ref3 < ${path_prefix}.bleu_cand

	#python ./tool/parent_data.py ${path_prefix}
	#python -m tool.PARENT.table_text_eval --references ${path_prefix}.parent_ref --generations ${path_prefix}.parent_pred --tables ${path_prefix}.parent_table --lambda_weight 0.5

	#python ./tool/nli_preprocess.py ${path_prefix}
	#python ./tool/nli_eval.py --type webnlg ${path_prefix}.nli ./tool/output.json
	#python tool/nli_postprocess.py
done
echo ''


################
# FNN
################

lang_list=(br cy ga ru)
test_from_list=(4500 3000 4500 4000)
for i in "${!lang_list[@]}"
do
	lang=${lang_list[i]}
        test_from=${test_from_list[i]}
	base_path=/rds/user/hpcxu1/hpc-work/outputs.webnlg/${lang}triple.full/
	if [ "$test_type" = "seen" ]; then
		path_prefix=${base_path}/logs.re.nn.ffn_nosample/test.res.${test_from}
	else
		path_prefix=${base_path}/logs.re.nn.ffn_nosample/test_unseen.res.${test_from}
	fi

	echo '['${test_type}' FNN]; lang='${lang}
	python ./tool/evaluate_d2t_bleu.py ${path_prefix}
	./tool/multi-bleu.perl ${path_prefix}.bleu_ref1 ${path_prefix}.bleu_ref2 ${path_prefix}.bleu_ref3 < ${path_prefix}.bleu_cand

	#python ./tool/parent_data.py ${path_prefix}
	#python -m tool.PARENT.table_text_eval --references ${path_prefix}.parent_ref --generations ${path_prefix}.parent_pred --tables ${path_prefix}.parent_table --lambda_weight 0.5

	#python ./tool/nli_preprocess.py ${path_prefix}
	#python ./tool/nli_eval.py --type webnlg ${path_prefix}.nli ./tool/output.json
	#python tool/nli_postprocess.py
done
echo ''
'''
