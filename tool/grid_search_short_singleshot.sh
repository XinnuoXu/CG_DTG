#for i in $(seq 0.0 .05 1.0)

'''
log_path=/rds/user/hpcxu1/hpc-work/outputs.webnlg/short_single.logs.re.discriministic/test.res.2000
for i in $(seq 1 1 15)
do
	sh ./scripts_d2t.less_triple_single_shot/test_deterministic.sh $i
	echo "====== test_graph_selection_threshold: $i ======"
	sh ./tool/multi-bleu.sh $log_path
	python ./pyrouge/read_clusters_multiref.py "$log_path.cluster"
	echo ""
done

log_path=/rds/user/hpcxu1/hpc-work/outputs.webnlg/short_single.logs.re.nn/test.res.2000
for i in $(seq 0.0 .005 0.35)
do
	sh ./scripts_d2t.less_triple_single_shot/test_nn.sh $i
	echo "====== test_graph_selection_threshold: $i ======"
	sh ./tool/multi-bleu.sh $log_path
	python ./pyrouge/read_clusters_multiref.py "$log_path.cluster"
	echo ""
done
'''

log_path=/rds/user/hpcxu1/hpc-work/outputs.webnlg/short_single.logs.re.nn.spectral/test.res.5000
for i in $(seq 0.0 .005 0.35)
do
	sh ./scripts_d2t.less_triple_single_shot/test_reinforce_graph_nn.sh $i
	echo "====== test_graph_selection_threshold: $i ======"
	sh ./tool/multi-bleu.sh $log_path
	python ./pyrouge/read_clusters_multiref.py "$log_path.cluster"
	echo ""
done

'''
log_path=/rds/user/hpcxu1/hpc-work/outputs.webnlg/short_single.logs.re.nn.spectral_with_sample/test.res.4500
for i in $(seq 0.0 .005 0.35)
do
	sh ./scripts_d2t.less_triple_single_shot/test_reinforce_graph_nn_sample.sh $i
	echo "====== test_graph_selection_threshold: $i ======"
	sh ./tool/multi-bleu.sh $log_path
	python ./pyrouge/read_clusters_multiref.py "$log_path.cluster"
	echo ""
done
'''
