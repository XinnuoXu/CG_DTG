ntriple=$1 #[2,3,4,7]
tokenizer=$2 #[t5-small, t5-base, t5-large]
test_from=$3
test_unseen=false

log_path=/rds/user/hpcxu1/hpc-work/outputs.webnlg/${ntriple}triple.full/logs.re.nn.randombase_sample.${tokenizer}/test.res.${test_from}
for i in $(seq 0.0 .005 0.6)
do
	sh ./scripts_d2t.ntriple_full/test_reinforce_graph_randombase_sample.sh ${ntriple} ${tokenizer} ${test_from} ${test_unseen} ${i}
	echo "====== test_graph_selection_threshold: $i ======"
	sh ./tool/multi-bleu.sh $log_path
	python ./pyrouge/read_clusters_multiref.py "$log_path.cluster"
	echo ""
done

