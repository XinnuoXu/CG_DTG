percent=$1
test_from=$2

log_path=/rds/user/hpcxu1/hpc-work/outputs.webnlg/webnlg_percent_${percent}/logs.re.discriministic/test.res.${test_from}
for i in $(seq 0 1 25)
do
	sh ./scripts_d2t.fewshot/test_deterministic.sh ${percent} ${test_from} $i
	echo "====== test_graph_selection_threshold: $i ======"
	sh ./tool/multi-bleu.sh $log_path
	#python ./pyrouge/read_clusters_multiref.py "$log_path.cluster"
	echo ""
done
