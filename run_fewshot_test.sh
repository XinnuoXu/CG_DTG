'''
sh ./scripts_d2t.fewshot/test_reinforce_encdec_partial.sh 0.005 500
sh ./scripts_d2t.fewshot/test_reinforce_encdec_partial.sh 0.01 500
sh ./scripts_d2t.fewshot/test_reinforce_encdec_partial.sh 0.05 2500
sh ./scripts_d2t.fewshot/test_reinforce_encdec_partial.sh 0.1 4500
'''


sh ./scripts_d2t.fewshot/test_deterministic.sh 0.005 500 2
sh ./scripts_d2t.fewshot/test_deterministic.sh 0.01 500 1
sh ./scripts_d2t.fewshot/test_deterministic.sh 0.05 2500 2
sh ./scripts_d2t.fewshot/test_deterministic.sh 0.1 4500 1


sh ./scripts_d2t.fewshot/test_nn.sh 0.005 500 0.2 
sh ./scripts_d2t.fewshot/test_nn.sh 0.01 500 0.3
sh ./scripts_d2t.fewshot/test_nn.sh 0.05 500 0.25
sh ./scripts_d2t.fewshot/test_nn.sh 0.1 1000 0.1


sh ./scripts_d2t.fewshot/test_random.sh 0.005 500
sh ./scripts_d2t.fewshot/test_random.sh 0.01 500
sh ./scripts_d2t.fewshot/test_random.sh 0.05 2500
sh ./scripts_d2t.fewshot/test_random.sh 0.1 4500
