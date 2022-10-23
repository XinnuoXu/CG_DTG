# Full set

#python pyrouge/run_for_ama.py \
#	-gold_type raw \
#	-gold_path ../Plan_while_Generate/AmaSum/AmaSum_data/test.jsonl \
#	-cand_type selsum \
#	-cand_path ../Plan_while_Generate/AmaSum/SelSum \

#python pyrouge/run_for_ama.py \
#	-gold_type raw \
#	-gold_path ../Plan_while_Generate/AmaSum/AmaSum_data/test.jsonl \
#	-cand_type systems \
#	-cand_path ./outputs.ama/logs.summarizer/test.res.300000.candidate \
#	-eid_path ./outputs.ama/logs.summarizer/test.res.300000.eid \

#python pyrouge/run_for_ama.py \
#	-gold_type raw \
#	-gold_path ../Plan_while_Generate/AmaSum/AmaSum_data/test.jsonl \
#	-cand_type systems \
#	-cand_path ./outputs.ama/logs.summarizer.v2/test.res.100000.candidate \
#	-eid_path ./outputs.ama/logs.summarizer.v2/test.res.100000.eid \

#############################################################################################
# Full clean set

#python pyrouge/run_for_ama.py \
#	-gold_type clean \
#	-gold_path ./outputs.ama/cleaned_testset/ \
#	-cand_type selsum \
#	-cand_path ../Plan_while_Generate/AmaSum/SelSum \

#python pyrouge/run_for_ama.py \
#	-gold_type clean \
#	-gold_path ./outputs.ama/cleaned_testset/ \
#	-cand_type systems \
#	-cand_path ./outputs.ama/logs.summarizer.longer/test.res.100000.candidate \
#	-eid_path ./outputs.ama/logs.summarizer.longer/test.res.100000.eid \

#python pyrouge/run_for_ama.py \
#	-gold_type clean \
#	-gold_path ./outputs.ama/cleaned_testset/ \
#	-cand_type systems \
#	-cand_path ./outputs.ama/logs.summarizer.v3.pro_or_con/test.res.100000.candidate \
#	-eid_path ./outputs.ama/logs.summarizer.v3.pro_or_con/test.res.100000.eid \

#############################################################################################
# Freq=3/10

#python pyrouge/run_for_ama.py \
#	-gold_type clean \
#	-gold_path ./outputs.ama/cleaned_testset.min10/ \
#	-cand_type selsum \
#	-cand_path ../Plan_while_Generate/AmaSum/SelSum \

#python pyrouge/run_for_ama.py \
#	-gold_type clean \
#	-gold_path ./outputs.ama/cleaned_testset.min10/ \
#	-cand_type systems \
#	-cand_path ./outputs.ama/logs.summarizer.sentence_level/test.res.100000.candidate \
#	-eid_path ./outputs.ama/logs.summarizer.sentence_level/test.res.100000.eid \

#python pyrouge/run_for_ama.py \
#	-gold_type clean \
#	-gold_path ./outputs.ama/cleaned_testset.min10/ \
#	-cand_type systems \
#	-cand_path ./outputs.ama/logs.summarizer.v2/test.res.100000.candidate \
#	-eid_path ./outputs.ama/logs.summarizer.v2/test.res.100000.eid \

python pyrouge/run_for_ama.py \
	-gold_type clean \
	-gold_path ./outputs.ama/cleaned_testset.min10/ \
	-cand_type systems \
	-cand_path ./outputs.ama/logs.summarizer.v3/test.res.100000.candidate \
	-eid_path ./outputs.ama/logs.summarizer.v3/test.res.100000.eid \

#python pyrouge/run_for_ama.py \
#	-gold_type clean \
#	-gold_path ./outputs.ama/cleaned_testset.min10/ \
#	-cand_type systems \
#	-cand_path ./outputs.ama/logs.summarizer.v3.pro_or_con/test.res.100000.candidate \
#	-eid_path ./outputs.ama/logs.summarizer.v3.pro_or_con/test.res.100000.eid \

#############################################################################################


#python pyrouge/run_for_ama.py \
#	-gold_type cluster_clean \
#	-gold_path ./outputs.ama/logs.summarizer/test.res.100000.gold \
#	-eid_path ./outputs.ama/logs.summarizer/test.res.100000.eid \
#	-cand_type selsum \
#	-cand_path ../Plan_while_Generate/AmaSum/SelSum \

#python pyrouge/run_for_ama.py \
#	-gold_type cluster_clean \
#	-gold_path ./outputs.ama/seq2seq/ \
#	-eid_path ./outputs.ama/logs.summarizer/test.res.100000.eid \
#	-cand_type systems \
#	-cand_path ./outputs.ama/logs.summarizer/test.res.100000.candidate \

