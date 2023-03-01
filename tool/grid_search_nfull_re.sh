ntriple=$1 #[2,3,4,7]
tokenizer=$2 #[t5-small, t5-base, t5-large]
test_from=$3
test_unseen=false

'''
log_path=/rds/user/hpcxu1/hpc-work/outputs.webnlg/${ntriple}triple.full/logs.re.nn.spectral.${tokenizer}/test.res.${test_from}
#for i in $(seq 0.0 .005 0.3)
for i in $(seq 0.0 .01 1.0)
do
	sh ./scripts_d2t.ntriple_full/test_reinforce_graph_nn.sh ${ntriple} ${tokenizer} ${test_from} ${test_unseen} ${i}
	echo "====== test_graph_selection_threshold: $i ======"
	sh ./tool/multi-bleu.sh $log_path
	python ./pyrouge/read_clusters_multiref.py "$log_path.cluster"
	echo ""
done

log_path=/rds/user/hpcxu1/hpc-work/outputs.webnlg/${ntriple}triple.full/logs.re.nn.threshold.${tokenizer}/test.res.${test_from}
#for i in $(seq 0.05 .01 0.5)
for i in $(seq 0.0 .01 1.0)
do
	sh ./scripts_d2t.ntriple_full/test_reinforce_graph_nn_adja_threshold.sh ${ntriple} ${tokenizer} ${test_from} ${test_unseen} ${i}
	echo "====== test_graph_selection_threshold: $i ======"
	sh ./tool/multi-bleu.sh $log_path
	python ./pyrouge/read_clusters_multiref.py "$log_path.cluster"
	echo ""
done

'''

'''
# Run bernoulli
log_path=/rds/user/hpcxu1/hpc-work/outputs.webnlg/${ntriple}triple.full/logs.re.nn.strongbase_sample.${tokenizer}/test.res.${test_from}
#for i in $(seq 0.05 .05 0.2)
for i in $(seq 0.0 .005 0.25)
do
	sh ./scripts_d2t.ntriple_full/test_reinforce_graph_nn_sample.sh ${ntriple} ${tokenizer} ${test_from} ${test_unseen} ${i}
	echo "====== test_graph_selection_threshold: $i ======"
	sh ./tool/multi-bleu.sh $log_path
	python ./pyrouge/read_clusters_multiref.py "$log_path.cluster"
	echo ""
done

# No bernoulli
log_path=/rds/user/hpcxu1/hpc-work/outputs.webnlg/${ntriple}triple.full/logs.re.nn.strongbase.${tokenizer}/test.res.${test_from}
#for i in $(seq 0.0 .1 1.0)
for i in $(seq 0.0 .005 0.25)
do
	sh ./scripts_d2t.ntriple_full/test_reinforce_graph_nn.sh ${ntriple} ${tokenizer} ${test_from} ${test_unseen} ${i}
	echo "====== test_graph_selection_threshold: $i ======"
	sh ./tool/multi-bleu.sh $log_path
	python ./pyrouge/read_clusters_multiref.py "$log_path.cluster"
	echo ""
done
'''

# No bernoulli, but with threshold
log_path=/rds/user/hpcxu1/hpc-work/outputs.webnlg/${ntriple}triple.full/logs.re.nn.strongbase_threshold.${tokenizer}/test.res.${test_from}
for i in $(seq 0.0 .005 0.25)
do
	sh ./scripts_d2t.ntriple_full/test_reinforce_graph_nn_adja_threshold.sh ${ntriple} ${tokenizer} ${test_from} ${test_unseen} ${i}
	echo "====== test_graph_selection_threshold: $i ======"
	sh ./tool/multi-bleu.sh $log_path
	python ./pyrouge/read_clusters_multiref.py "$log_path.cluster"
	echo ""
done
