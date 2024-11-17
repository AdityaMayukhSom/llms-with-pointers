python -m rouge.rouge `
    --target_filepattern=./results_test/*_gen_abstract.txt `
    --prediction_filepattern=./results_test/*_ori_abstract.txt `
    --output_filename=scores.csv `
    --use_stemmer=true `
    --split_summaries=true