#/bin/bash

path_prefix='/rds/user/hpcxu1/hpc-work/outputs.webnlg/logs.base/test.res.9000'
#path_prefix='/rds/user/hpcxu1/hpc-work/outputs.webnlg/logs.re.base/test.res.7000'
#path_prefix='/rds/user/hpcxu1/hpc-work/outputs.webnlg/logs.re.encdec_partial_numerical/test.res.7000'
#path_prefix='/rds/user/hpcxu1/hpc-work/outputs.webnlg/logs.re.encdec_partial/test.res.7000'

#path_prefix='/rds/user/hpcxu1/hpc-work/outputs.webnlg/logs.re.random/test.res.7000'
#path_prefix='/rds/user/hpcxu1/hpc-work/outputs.webnlg/logs.re.discriministic/test.res.7000'
#path_prefix='/rds/user/hpcxu1/hpc-work/outputs.webnlg/logs.re.from_scratch/test.res.9000'
#path_prefix='/rds/user/hpcxu1/hpc-work/outputs.webnlg/logs.nn/test.res.4000'
#path_prefix='/rds/user/hpcxu1/hpc-work/outputs.webnlg/logs.re.nn.spectral/test.res.6000'
#path_prefix='/rds/user/hpcxu1/hpc-work/outputs.webnlg/logs.re.nn.spectral_with_sample//test.res.6000'
#path_prefix=$1

python ./tool/evaluate_d2t_bleu.py ${path_prefix}

./tool/multi-bleu.perl ${path_prefix}.bleu_ref1 ${path_prefix}.bleu_ref2 ${path_prefix}.bleu_ref3 < ${path_prefix}.bleu_cand

