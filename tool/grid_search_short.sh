#for i in $(seq 0.0 .05 1.0)

'''
log_path=/rds/user/hpcxu1/hpc-work/outputs.webnlg/short.logs.re.discriministic/test.res.5000
for i in $(seq 2 2 30)
do
	sh ./scripts_d2t.train_less_triple/test_deterministic.sh $i
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

#log_path=/rds/user/hpcxu1/hpc-work/outputs.webnlg/logs.nn/test.res.10000
log_path=/rds/user/hpcxu1/hpc-work/outputs.webnlg/short.logs.re.nn/test.res.4000
for i in $(seq 0.0 .005 0.35)
do
	sh ./scripts_d2t.train_less_triple/test_nn.sh $i
	echo "====== test_graph_selection_threshold: $i ======"
	sh ./tool/multi-bleu.sh $log_path
	python ./pyrouge/read_clusters_multiref.py "$log_path.cluster"
	echo ""
done

#log_path=/rds/user/hpcxu1/hpc-work/outputs.webnlg/logs.re.nn/test.res.15000
#log_path=/rds/user/hpcxu1/hpc-work/outputs.webnlg/logs.re.nn.spectral/test.res.6000
log_path=/rds/user/hpcxu1/hpc-work/outputs.webnlg/short.logs.re.nn.spectral/test.res.5000
for i in $(seq 0.1 .005 0.35)
do
	sh ./scripts_d2t.train_less_triple/test_reinforce_graph_nn.sh $i
	echo "====== test_graph_selection_threshold: $i ======"
	sh ./tool/multi-bleu.sh $log_path
	python ./pyrouge/read_clusters_multiref.py "$log_path.cluster"
	echo ""
done

'''

#log_path=/rds/user/hpcxu1/hpc-work/outputs.webnlg/logs.re.nn.sample/test.res.15000
#log_path=/rds/user/hpcxu1/hpc-work/outputs.webnlg/logs.re.nn.spectral_with_sample/test.res.6000
log_path=/rds/user/hpcxu1/hpc-work/outputs.webnlg/short.logs.re.nn.spectral_with_sample/test.res.6000
for i in $(seq 0.1 .005 0.35)
do
	sh ./scripts_d2t.train_less_triple/test_reinforce_graph_nn_sample.sh $i
	echo "====== test_graph_selection_threshold: $i ======"
	sh ./tool/multi-bleu.sh $log_path
	python ./pyrouge/read_clusters_multiref.py "$log_path.cluster"
	echo ""
done
