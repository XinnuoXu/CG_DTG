ntriple=$1
test_from=$2
test_unseen=false

log_path=/rds/user/hpcxu1/hpc-work/outputs.webnlg/${ntriple}triple.single/short_single.logs.re.discriministic//test.res.${test_from}
for i in $(seq 0 1 30)
do
	sh ./scripts_d2t.ntriple_single/test_deterministic.sh ${ntriple} ${test_from} ${test_unseen} $i
	echo "====== test_graph_selection_threshold: $i ======"
	sh ./tool/multi-bleu.sh $log_path
	python ./pyrouge/read_clusters_multiref.py "$log_path.cluster"
	echo ""
done

