#/bin/bash

#path_prefix='/rds/user/hpcxu1/hpc-work/outputs.webnlg/logs.base/test.res.3000'
#path_prefix='/rds/user/hpcxu1/hpc-work/outputs.webnlg/logs.re.base/test.res.2000'
#path_prefix='/rds/user/hpcxu1/hpc-work/outputs.webnlg/logs.re.from_scratch/test.res.40000'
#path_prefix='/rds/user/hpcxu1/hpc-work/outputs.webnlg/logs.re.gold_random/test.res.20000'
#path_prefix='/rds/user/hpcxu1/hpc-work/outputs.webnlg/logs.re.evenly_mix/test.res.20000'
path_prefix='/rds/user/hpcxu1/hpc-work/outputs.webnlg/logs.re.joint/test.res.35000'

python ./tool/evaluate_d2t_bleu.py ${path_prefix}

./tool/multi-bleu.perl ${path_prefix}.bleu_ref1 ${path_prefix}.bleu_ref2 ${path_prefix}.bleu_ref3 < ${path_prefix}.bleu_cand

