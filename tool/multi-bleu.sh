#/bin/bash

#path_prefix='./outputs.webnlg/logs.step/test.res.4000'
path_prefix='./outputs.webnlg/logs.base/test.res'
#path_prefix='./outputs.webnlg/logs.tree_during_infer/test.res.4000'
target_format='first-to-many'

python ./tool/evaluate_d2t_bleu.py ${path_prefix} ${target_format}

if [ ${target_format} = 'first-to-many' ]
then
    ./tool/multi-bleu.perl ${path_prefix}.bleu_ref1 ${path_prefix}.bleu_ref2 ${path_prefix}.bleu_ref3 < ${path_prefix}.bleu_cand
fi

if [ ${target_format} = 'first-to-first' ]
then
    ./tool/multi-bleu.perl ${path_prefix}.bleu_ref1 < ${path_prefix}.bleu_cand
fi
