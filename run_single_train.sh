#sh tool/grid_search_nshort_randombase_nosample.sh 2 1000 > ./tool/grid.res.ntriples.single_new/randombase_nosample_2.res
#sh tool/grid_search_nshort_randombase_nosample.sh 3 3000 > ./tool/grid.res.ntriples.single_new/randombase_nosample_3.res
#sh tool/grid_search_nshort_randombase_nosample.sh 4 8000 > ./tool/grid.res.ntriples.single_new/randombase_nosample_4.res
#sh tool/grid_search_nshort_randombase_nosample.sh 5 8000 > ./tool/grid.res.ntriples.single_new/randombase_nosample_5.res
#sh tool/grid_search_nshort_randombase_nosample.sh 6 3000 > ./tool/grid.res.ntriples.single_new/randombase_nosample_6.res
#sh tool/grid_search_nshort_randombase_nosample.sh 7 6000 > ./tool/grid.res.ntriples.single_new/randombase_nosample_7.res

sh ./scripts_d2t.ntriple_single/train_nn.sh 2 2000
sh ./scripts_d2t.ntriple_single/train_nn.sh 3 3000
sh ./scripts_d2t.ntriple_single/train_nn.sh 4 4000

#sh ./scripts_d2t.ntriple_single/train_nn.sh 5 5000
#sh ./scripts_d2t.ntriple_single/train_nn.sh 6 6000
#sh ./scripts_d2t.ntriple_single/train_nn.sh 7 8000
