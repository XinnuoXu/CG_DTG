percent=$1
test_from=$2

log_path=/rds/user/hpcxu1/hpc-work/outputs.webnlg/webnlg_percent_${percent}/logs.re.nn.spectral//test.res.${test_from}
for i in $(seq 0.00 .01 0.5)
#for i in $(seq 0.0 .05 0.35)
do
	sh ./scripts_d2t.fewshot/test_reinforce_graph_nn.sh ${percent} ${test_from} $i
	echo "====== test_graph_selection_threshold: $i ======"
	sh ./tool/multi-bleu.sh $log_path
	#python ./pyrouge/read_clusters_multiref.py "$log_path.cluster"
	echo ""
done

