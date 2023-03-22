#sh ./scripts_d2t.ntriple_full/train_reinforce_graph_nn.sh 7 t5-small 4000
#sh ./scripts_d2t.ntriple_full/train_reinforce_graph_nn.sh 4 t5-small 4000
#sh ./scripts_d2t.ntriple_full/train_reinforce_graph_nn.sh 3 t5-small 4000
#sh ./scripts_d2t.ntriple_full/train_reinforce_graph_nn.sh 2 t5-small 4000

#sh ./scripts_d2t.ntriple_single/train_reinforce_graph_nn.sh 7 3000
#sh ./scripts_d2t.ntriple_single/train_reinforce_graph_nn.sh 6 2000
#sh ./scripts_d2t.ntriple_single/train_reinforce_graph_nn.sh 5 3000
#sh ./scripts_d2t.ntriple_single/train_reinforce_graph_nn.sh 4 2000
#sh ./scripts_d2t.ntriple_single/train_reinforce_graph_nn.sh 3 2000
#sh ./scripts_d2t.ntriple_single/train_reinforce_graph_nn.sh 2 2000

#sh ./scripts_d2t.fewshot/train_reinforce_encdec_partial.sh 0.1
#sh ./scripts_d2t.fewshot/train_reinforce_encdec_partial.sh 0.01
#sh ./scripts_d2t.fewshot/train_reinforce_encdec_partial.sh 0.05
#sh ./scripts_d2t.fewshot/train_reinforce_encdec_partial.sh 0.005

sh ./scripts_d2t.fewshot/train_nn.sh 0.005 500
sh ./scripts_d2t.fewshot/train_nn.sh 0.01 500
sh ./scripts_d2t.fewshot/train_nn.sh 0.05 2500
sh ./scripts_d2t.fewshot/train_nn.sh 0.005 3000
