#sh ./scripts_d2t.fewshot/train_reinforce_encdec_partial.sh 0.1
#sh ./scripts_d2t.fewshot/train_reinforce_encdec_partial.sh 0.01
#sh ./scripts_d2t.fewshot/train_reinforce_encdec_partial.sh 0.05
#sh ./scripts_d2t.fewshot/train_reinforce_encdec_partial.sh 0.005

sh ./scripts_d2t.fewshot/train_nn.sh 0.1 4500
sh ./scripts_d2t.fewshot/train_nn.sh 0.05 2500
sh ./scripts_d2t.fewshot/train_nn.sh 0.01 500
sh ./scripts_d2t.fewshot/train_nn.sh 0.005 500
