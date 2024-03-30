result_dir=eval_logs
rm -rf $result_dir
mkdir -p $result_dir
jupyter nbconvert --to script snlp_test.ipynb
python3 snlp_test.py &> $result_dir/mteb_eval_logs.txt 2>&1 &