#coding=utf8

from collections import Counter, OrderedDict

base_path = './outputs.webnlg/logs.plan_generation/test.res.4000'

if __name__ == '__main__':
    
    src_path = base_path + '.raw_src'
    cand_path = base_path + '.candidate'
    gold_path = base_path + '.gold'

    input_preds = []
    for line in open(src_path):
        line = line.strip()
        triples = line.split('<PRED>')[1:]
        preds = [triple.split('<OBJ>')[0] for triple in triples]
        input_preds.append(preds)

    output_preds = []
    for line in open(cand_path):
        line = line.strip()
        flist = line.split(' ')
        output_preds.append(flist)

    golds = []
    for line in open(gold_path):
        line = line.strip()
        flist = line.split('<ref-sep> ')[1:]
        golds.append(flist)

    extrinsic_hallucination_example_num = 0
    omission_example_num = 0
    wrong_length_num = 0
    wrong_repeat_num = 0
    bad_example_num = 0
    example_num = 0
    ref_chunk_lengh = []
    cand_chunk_lengh = []

    for pair in list(zip(input_preds, output_preds, golds)):
        input_pred = pair[0]
        output_pred = pair[1]
        gold_pred = pair[2]

        #if len(input_pred) < 3:
        #    continue
        example_num += 1

        extrinsic_hallucination_example = False
        for tok in output_pred:
            if tok == '|||':
                continue
            if tok not in input_pred:
                extrinsic_hallucination_example = True
        if extrinsic_hallucination_example:
            extrinsic_hallucination_example_num += 1

        omission_example = False
        for tok in input_pred:
            if tok not in output_pred:
                omission_example = True
        if omission_example:
            omission_example_num += 1

        wrong_length_example = False
        if len(input_pred) != len([tok for tok in output_pred if tok != '|||']):
            wrong_length_num += 1
            wrong_length_example = True

        wrong_repeat_example = False
        for tok in output_pred:
            if tok == '|||':
                continue
            if tok not in input_pred:
                continue
            output_side = output_pred.count(tok)
            input_side = input_pred.count(tok)
            if output_side > 1 and output_side > input_side:
                wrong_repeat_example = True
        if wrong_repeat_example:
            wrong_repeat_num += 1

        if extrinsic_hallucination_example or omission_example or wrong_repeat_example or wrong_length_example:
            bad_example_num += 1

        for chunk in ' '.join(output_pred).split(' ||| '):
            cand_chunk_lengh.append(len(chunk.split(' ')))

        for one_ref in gold_pred:
            for chunk in one_ref.split(' ||| '):
                ref_chunk_lengh.append(len(chunk.split(' ')))

    print ('extrinsic_hallucination_example_num:', extrinsic_hallucination_example_num) 
    print ('omission_example_num:', omission_example_num)
    print ('wrong_repeat_num:', wrong_repeat_num)
    print ('bad_example_num:', bad_example_num)
    print ('total num of examples:', example_num)

    print ('avg chunk length in refs: ', sum(ref_chunk_lengh)/float(len(ref_chunk_lengh)))
    print ('avg chunk length in cands: ', sum(cand_chunk_lengh)/float(len(cand_chunk_lengh)))

    print ('detailed chunk length in refs: ', sorted(dict(Counter(ref_chunk_lengh)).items()))
    print ('detailed chunk length in cands: ', sorted(dict(Counter(cand_chunk_lengh)).items()))
