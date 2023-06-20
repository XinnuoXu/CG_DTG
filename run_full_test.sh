'''
sh ./scripts_d2t.ntriple_full/test_reinforce_encdec_partial.sh 2 t5-base 10000 false
sh ./scripts_d2t.ntriple_full/test_reinforce_encdec_partial.sh 3 t5-base 10000 false
sh ./scripts_d2t.ntriple_full/test_reinforce_encdec_partial.sh 4 t5-base 15000 false
sh ./scripts_d2t.ntriple_full/test_reinforce_encdec_partial.sh 7 t5-base 30000 false

sh ./scripts_d2t.ntriple_full/test_reinforce_encdec_partial.sh 2 t5-base 10000 true
sh ./scripts_d2t.ntriple_full/test_reinforce_encdec_partial.sh 3 t5-base 10000 true
sh ./scripts_d2t.ntriple_full/test_reinforce_encdec_partial.sh 4 t5-base 15000 true
sh ./scripts_d2t.ntriple_full/test_reinforce_encdec_partial.sh 7 t5-base 30000 true



sh ./scripts_d2t.ntriple_full/test_deterministic.sh 2 t5-base 10000 false 1
sh ./scripts_d2t.ntriple_full/test_deterministic.sh 3 t5-base 10000 false 1
sh ./scripts_d2t.ntriple_full/test_deterministic.sh 4 t5-base 15000 false 1
sh ./scripts_d2t.ntriple_full/test_deterministic.sh 7 t5-base 30000 false 1

sh ./scripts_d2t.ntriple_full/test_deterministic.sh 2 t5-base 10000 true 1
sh ./scripts_d2t.ntriple_full/test_deterministic.sh 3 t5-base 10000 true 1
sh ./scripts_d2t.ntriple_full/test_deterministic.sh 4 t5-base 15000 true 1
sh ./scripts_d2t.ntriple_full/test_deterministic.sh 7 t5-base 30000 true 1



sh ./scripts_d2t.ntriple_full/test_random.sh 2 t5-base 10000 false
sh ./scripts_d2t.ntriple_full/test_random.sh 3 t5-base 10000 false
sh ./scripts_d2t.ntriple_full/test_random.sh 4 t5-base 15000 false
sh ./scripts_d2t.ntriple_full/test_random.sh 7 t5-base 30000 false

sh ./scripts_d2t.ntriple_full/test_random.sh 2 t5-base 10000 true
sh ./scripts_d2t.ntriple_full/test_random.sh 3 t5-base 10000 true
sh ./scripts_d2t.ntriple_full/test_random.sh 4 t5-base 15000 true
sh ./scripts_d2t.ntriple_full/test_random.sh 7 t5-base 30000 true



sh ./scripts_d2t.ntriple_full/test_ffn_nosample.sh 2 t5-base 500 false 0.02
sh ./scripts_d2t.ntriple_full/test_ffn_nosample.sh 3 t5-base 1500 false 0.1
sh ./scripts_d2t.ntriple_full/test_ffn_nosample.sh 4 t5-base 2000 false 0.02
sh ./scripts_d2t.ntriple_full/test_ffn_nosample.sh 7 t5-base 5500 false 0.01

sh ./scripts_d2t.ntriple_full/test_ffn_nosample.sh 2 t5-base 500 true 0.02
sh ./scripts_d2t.ntriple_full/test_ffn_nosample.sh 3 t5-base 1500 true 0.1
sh ./scripts_d2t.ntriple_full/test_ffn_nosample.sh 4 t5-base 2000 true 0.02
sh ./scripts_d2t.ntriple_full/test_ffn_nosample.sh 7 t5-base 5500 true 0.01
'''




sh ./scripts_d2t.ntriple_full/test_reinforce_graph_randombase_nosample.sh 2 t5-base 2000 false 0.04
sh ./scripts_d2t.ntriple_full/test_reinforce_graph_randombase_nosample.sh 3 t5-base 7000 false 0.5
sh ./scripts_d2t.ntriple_full/test_reinforce_graph_randombase_nosample.sh 4 t5-base 5000 false 0.30
sh ./scripts_d2t.ntriple_full/test_reinforce_graph_randombase_nosample.sh 7 t5-base 4000 false 0.005

sh ./scripts_d2t.ntriple_full/test_reinforce_graph_randombase_nosample.sh 2 t5-base 2000 true 0.04
sh ./scripts_d2t.ntriple_full/test_reinforce_graph_randombase_nosample.sh 3 t5-base 7000 true 0.5
sh ./scripts_d2t.ntriple_full/test_reinforce_graph_randombase_nosample.sh 4 t5-base 5000 true 0.30
sh ./scripts_d2t.ntriple_full/test_reinforce_graph_randombase_nosample.sh 7 t5-base 4000 true 0.005
