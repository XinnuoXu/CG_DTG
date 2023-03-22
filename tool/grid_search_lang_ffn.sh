lang=$1 #[br, cy, ga, ru]
test_from=$2

log_path=/rds/user/hpcxu1/hpc-work/outputs.webnlg/${lang}triple.full/logs.re.nn.ffn_nosample/test.res.${test_from}
for i in $(seq 0.0 0.02 0.4)
do
	sh ./scripts_d2t.lang/test_ffn_nosample.sh  ${lang} ${test_from} ${i}
	echo "====== test_graph_selection_threshold: $i ======"
	sh ./tool/multi-bleu.sh $log_path
	#python ./pyrouge/read_clusters_multiref.py "$log_path.cluster"
	echo ""
done
