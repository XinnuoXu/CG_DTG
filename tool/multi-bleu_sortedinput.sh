#/bin/bash

#path_prefix='./outputs.webnlg/logs.base/test.res.3000'
#path_prefix='./outputs.webnlg/logs.src_prompt/test.res.4000'
#path_prefix='./outputs.webnlg/logs.tgt_prompt/test.res.5000'
path_prefix='./outputs.webnlg/logs.step_wise/test.res.6000'
#path_prefix='./outputs.webnlg/logs.agg_encoder/test.res.4000'

#target_format='one-to-one'
#target_format='one-to-many'
target_format='first-to-first'

python ./tool/evaluate_d2t_bleu_sortedinput.py ${path_prefix} ${target_format}

if [ ${target_format} = 'one-to-many' ]
then
    ./tool/multi-bleu.perl ${path_prefix}.bleu_ref1 ${path_prefix}.bleu_ref2 ${path_prefix}.bleu_ref3 < ${path_prefix}.bleu_cand
fi

if [ ${target_format} = 'first-to-first' ]
then
    ./tool/multi-bleu.perl ${path_prefix}.bleu_ref1 < ${path_prefix}.bleu_cand
fi

if [ ${target_format} = 'one-to-one' ]
then
    ./tool/multi-bleu.perl ${path_prefix}.ref < ${path_prefix}.cand
fi
