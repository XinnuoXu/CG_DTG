#/bin/bash

path_prefix='/rds/user/hpcxu1/hpc-work/outputs.webnlg/logs.base/test.res.10000'
#path_prefix='/rds/user/hpcxu1/hpc-work/outputs.webnlg/logs.re.base/test.res.10000'
#path_prefix='/rds/user/hpcxu1/hpc-work/outputs.webnlg/logs.re.encdec_partial_numerical/test.res.10000'
#path_prefix='/rds/user/hpcxu1/hpc-work/outputs.webnlg/logs.re.encdec_partial/test.res.10000'
#path_prefix='/rds/user/hpcxu1/hpc-work/outputs.webnlg/logs.re.discriministic/test.res.2000'
#path_prefix='/rds/user/hpcxu1/hpc-work/outputs.webnlg/logs.re.from_scratch/test.res.10000'
#path_prefix='/rds/user/hpcxu1/hpc-work/outputs.webnlg/logs.nn/test.res.10000'
#path_prefix='/rds/user/hpcxu1/hpc-work/outputs.webnlg/logs.re.nn//test.res.20000'
#path_prefix='/rds/user/hpcxu1/hpc-work/outputs.webnlg/logs.re.nn.sample/test.res.20000'
#path_prefix=$1

python ./tool/evaluate_d2t_bleu.py ${path_prefix}

./tool/multi-bleu.perl ${path_prefix}.bleu_ref1 ${path_prefix}.bleu_ref2 ${path_prefix}.bleu_ref3 < ${path_prefix}.bleu_cand

