
path_prefix=/rds/user/hpcxu1/hpc-work/outputs.webnlg/2triple.single//short_single.logs.re.nn.spectral_with_sample/test.res.2500

python ./tool/nli_preprocess.py ${path_prefix}
python ./tool/nli_eval.py --type webnlg ${path_prefix}.nli ./output.json
python tool/nli_postprocess.py
