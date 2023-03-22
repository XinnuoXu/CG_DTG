#sh tool/grid_search_nfull_determin.sh 2 t5-base 10000 > ./tool/grid.res.ntriples.full/grid.res.2.determin 
#sh tool/grid_search_nfull_determin.sh 3 t5-base 10000 > ./tool/grid.res.ntriples.full/grid.res.3.determin 
#sh tool/grid_search_nfull_determin.sh 4 t5-base 15000 > ./tool/grid.res.ntriples.full/grid.res.4.determin 
#sh tool/grid_search_nfull_determin.sh 7 t5-base 30000 > ./tool/grid.res.ntriples.full/grid.res.7.determin 

#sh tool/grid_search_nfull_ffn_nosample.sh 4 t5-base 2000 > ./tool/grid.res.ntriples.full/grid.res.4.nn
sh tool/grid_search_nfull_ffn_nosample.sh 7 t5-base 5500 > ./tool/grid.res.ntriples.full/grid.res.7.nn
sh tool/grid_search_nfull_ffn_nosample.sh 2 t5-base 500 > ./tool/grid.res.ntriples.full/grid.res.2.nn
sh tool/grid_search_nfull_ffn_nosample.sh 3 t5-base 1500 > ./tool/grid.res.ntriples.full/grid.res.3.nn
