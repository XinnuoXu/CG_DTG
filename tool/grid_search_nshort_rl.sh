ntriple=$1
test_from=$2

log_path=/rds/user/hpcxu1/hpc-work/outputs.webnlg/${ntriple}triple.single/short_single.logs.re.nn.spectral/test.res.${test_from}
for i in $(seq 0.1 .02 1.0)
#for i in $(seq 0.05 .01 0.38)
do
	sh ./scripts_d2t.ntriple_single/test_reinforce_graph_nn.sh ${ntriple} ${test_from} $i
	echo "====== test_graph_selection_threshold: $i ======"
	sh ./tool/multi-bleu.sh $log_path
	python ./pyrouge/read_clusters_multiref.py "$log_path.cluster"
	echo ""
done

