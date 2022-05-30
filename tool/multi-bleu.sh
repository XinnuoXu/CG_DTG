#/bin/bash

#path_prefix='./outputs.webnlg/logs.step/test.res.4000'
#path_prefix='./outputs.webnlg/logs.base/test.res'
#path_prefix='./outputs.webnlg/logs.tree_during_infer/test.res.4000'
#path_prefix='./outputs.webnlg/logs.plan/test.res.4000'
#path_prefix='./outputs.webnlg/logs.pred.base/test.res.4000'
#path_prefix='./outputs.webnlg//logs.pred.edge.discrete/test.res.5000'
#path_prefix='./outputs.webnlg//logs.pred.edge.discrete.bak/test.res.4000'
#path_prefix='./outputs.webnlg//logs.pred.edge.marginal/test.res.4000'
#path_prefix='outputs.webnlg//logs.pred.self.discrete/test.res.4000'
#path_prefix='outputs.webnlg//logs.pred.self.marginal/test.res.4000'

python ./tool/evaluate_d2t_bleu.py ${path_prefix}

./tool/multi-bleu.perl ${path_prefix}.bleu_ref1 ${path_prefix}.bleu_ref2 ${path_prefix}.bleu_ref3 < ${path_prefix}.bleu_cand

