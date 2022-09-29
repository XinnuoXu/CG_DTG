
#python pyrouge/run_for_ama.py \
#	-gold_type cluster_clean \
#	-gold_path ./outputs.ama/logs.summarizer/test.res.120000.gold \
#	-eid_path ./outputs.ama/logs.summarizer/test.res.120000.eid \
#	-cand_type selsum \
#	-cand_path ../Plan_while_Generate/AmaSum/SelSum \

python pyrouge/run_for_ama.py \
	-gold_type cluster_clean \
	-gold_path ./outputs.ama/seq2seq/ \
	-eid_path ./outputs.ama/logs.summarizer/test.res.120000.eid \
	-cand_type systems \
	-cand_path ./outputs.ama/logs.summarizer/test.res.120000.candidate \

#python pyrouge/run_for_ama.py \
#	-gold_type raw \
#	-gold_path ../Plan_while_Generate/AmaSum/AmaSum_data/test.jsonl \
#	-cand_type selsum \
#	-cand_path ../Plan_while_Generate/AmaSum/SelSum \

#python pyrouge/run_for_ama.py \
#	-gold_type raw \
#	-gold_path ../Plan_while_Generate/AmaSum/AmaSum_data/test.jsonl \
#	-cand_type systems \
#	-cand_path ./outputs.ama/logs.summarizer/test.res.120000.candidate \
#	-eid_path ./outputs.ama/logs.summarizer/test.res.120000.eid \

#python pyrouge/run_for_ama.py \
#	-gold_type clean \
#	-gold_path ./outputs.ama/cleaned_testset/ \
#	-cand_type selsum \
#	-cand_path ../Plan_while_Generate/AmaSum/SelSum \

#python pyrouge/run_for_ama.py \
#	-gold_type clean \
#	-gold_path ./outputs.ama/cleaned_testset/ \
#	-cand_type systems \
#	-cand_path ./outputs.ama/logs.summarizer/test.res.120000.candidate \
#	-eid_path ./outputs.ama/logs.summarizer/test.res.120000.eid \
