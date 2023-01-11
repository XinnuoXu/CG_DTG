#for i in $(seq 0.0 .05 1.0)

'''
log_path=/rds/user/hpcxu1/hpc-work/outputs.webnlg/logs.re.discriministic/test.res.2000
for i in $(seq 0 5 100)
do
	sh ./scripts_d2t.hpc/test_deterministic.sh $i
	echo "====== test_graph_selection_threshold: $i ======"
	sh ./tool/multi-bleu.sh $log_path
	python ./pyrouge/read_clusters_multiref.py "$log_path.cluster"
	echo ""
done

log_path=/rds/user/hpcxu1/hpc-work/outputs.webnlg/logs.re.from_scratch/test.res.10000
for i in $(seq 0.0 .05 1.0)
do
	sh ./scripts_d2t.hpc/test_reinforce.sh $i
	echo "====== test_graph_selection_threshold: $i ======"
	sh ./tool/multi-bleu.sh $log_path
	python ./pyrouge/read_clusters_multiref.py "$log_path.cluster"
	echo ""
done

log_path=/rds/user/hpcxu1/hpc-work/outputs.webnlg/logs.nn/test.res.10000
for i in $(seq 0.0 .05 1.0)
do
	sh ./scripts_d2t.hpc/test_nn.sh $i
	echo "====== test_graph_selection_threshold: $i ======"
	sh ./tool/multi-bleu.sh $log_path
	python ./pyrouge/read_clusters_multiref.py "$log_path.cluster"
	echo ""
done

log_path=/rds/user/hpcxu1/hpc-work/outputs.webnlg/logs.re.nn/test.res.15000
for i in $(seq 0.0 .05 1.0)
do
	sh ./scripts_d2t.hpc/test_reinforce_nn.sh $i
	echo "====== test_graph_selection_threshold: $i ======"
	sh ./tool/multi-bleu.sh $log_path
	python ./pyrouge/read_clusters_multiref.py "$log_path.cluster"
	echo ""
done

'''
log_path=/rds/user/hpcxu1/hpc-work/outputs.webnlg/logs.re.nn.sample/test.res.15000
for i in $(seq 0.0 .05 1.0)
do
	sh ./scripts_d2t.hpc/test_reinforce_nn_sample.sh $i
	echo "====== test_graph_selection_threshold: $i ======"
	sh ./tool/multi-bleu.sh $log_path
	python ./pyrouge/read_clusters_multiref.py "$log_path.cluster"
	echo ""
done
