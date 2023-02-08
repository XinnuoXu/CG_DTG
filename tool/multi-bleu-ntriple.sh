#/bin/bash

test_type=$1

################
# Baseline
################

echo 'BLEU for ['${test_type}' Baseline]'
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
	python ./tool/evaluate_d2t_bleu.py ${path_prefix}
	./tool/multi-bleu.perl ${path_prefix}.bleu_ref1 ${path_prefix}.bleu_ref2 ${path_prefix}.bleu_ref3 < ${path_prefix}.bleu_cand
done

################
# Random
################

echo 'BLEU for ['${test_type}' Random]'
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
	python ./tool/evaluate_d2t_bleu.py ${path_prefix}
	./tool/multi-bleu.perl ${path_prefix}.bleu_ref1 ${path_prefix}.bleu_ref2 ${path_prefix}.bleu_ref3 < ${path_prefix}.bleu_cand
done

################
# Numerical
################

echo 'BLEU for ['${test_type}' Numerical]'
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
	python ./tool/evaluate_d2t_bleu.py ${path_prefix}
	./tool/multi-bleu.perl ${path_prefix}.bleu_ref1 ${path_prefix}.bleu_ref2 ${path_prefix}.bleu_ref3 < ${path_prefix}.bleu_cand
done

################
# FFN
################

echo 'BLEU for ['${test_type}' FFN]'
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
	python ./tool/evaluate_d2t_bleu.py ${path_prefix}
	./tool/multi-bleu.perl ${path_prefix}.bleu_ref1 ${path_prefix}.bleu_ref2 ${path_prefix}.bleu_ref3 < ${path_prefix}.bleu_cand
done

################
# FFN Reinforce
################

echo 'BLEU for ['${test_type}' FFN Reinforce]'
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
	python ./tool/evaluate_d2t_bleu.py ${path_prefix}
	./tool/multi-bleu.perl ${path_prefix}.bleu_ref1 ${path_prefix}.bleu_ref2 ${path_prefix}.bleu_ref3 < ${path_prefix}.bleu_cand
done

################
# FFN Reinforce Sample
################

echo 'BLEU for ['${test_type}' FFN Reinforce Sample]'
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
	python ./tool/evaluate_d2t_bleu.py ${path_prefix}
	./tool/multi-bleu.perl ${path_prefix}.bleu_ref1 ${path_prefix}.bleu_ref2 ${path_prefix}.bleu_ref3 < ${path_prefix}.bleu_cand
done
