#sh tool/grid_search_nfull_randombase_nosample.sh 2 t5-small 4000 > tool/grid.res.ntriples.full_new/randombase_nosample_2.res
#sh tool/grid_search_nfull_randombase_nosample.sh 3 t5-small 6000 > tool/grid.res.ntriples.full_new/randombase_nosample_3.res
#sh tool/grid_search_nfull_randombase_nosample.sh 4 t5-small 3000 > tool/grid.res.ntriples.full_new/randombase_nosample_4.res
#sh tool/grid_search_nfull_randombase_nosample.sh 7 t5-small 6000 > tool/grid.res.ntriples.full_new/randombase_nosample_7.res

#sh ./scripts_d2t.ntriple_full/train_nn.sh 2 t5-base 10000
#sh ./scripts_d2t.ntriple_full/train_nn.sh 3 t5-base 10000
#sh ./scripts_d2t.ntriple_full/train_nn.sh 4 t5-base 15000
#sh ./scripts_d2t.ntriple_full/train_nn.sh 7 t5-base 30000

sh ./scripts_d2t.ntriple_full/train_reinforce_graph_nn.sh 7 t5-base 5500 
sh ./scripts_d2t.ntriple_full/train_reinforce_graph_nn.sh 4 t5-base 2000
sh ./scripts_d2t.ntriple_full/train_reinforce_graph_nn.sh 3 t5-base 1500 
sh ./scripts_d2t.ntriple_full/train_reinforce_graph_nn.sh 2 t5-base 500 
