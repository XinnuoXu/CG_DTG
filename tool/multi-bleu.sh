#/bin/bash

#path_prefix='./outputs.webnlg/logs.base/test.res.3000'
path_prefix='./outputs.webnlg/logs.partial_src/test.res'
#path_prefix='./outputs.webnlg/logs.prefix_tgt/test.res.4000'
#path_prefix='./slogs/test.res'

python ./tool/evaluate_d2t_bleu.py ${path_prefix}

./tool/multi-bleu.perl ${path_prefix}.bleu_ref1 ${path_prefix}.bleu_ref2 ${path_prefix}.bleu_ref3 < ${path_prefix}.bleu_cand

