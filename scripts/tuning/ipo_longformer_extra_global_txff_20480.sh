# Experiment's settings
GPU_NUMBER=3
MODEL_NAME='ipo-longformer-extra-global'
ACCUMULATION_STEPS=1
TASK='IPO'
TRAIN_DATASET='/media/SSD_2TB/financial_data/financial_textual_dataset/balanced/risk/dataset-risk-train.json'
EVAL_DATASET='/media/SSD_2TB/financial_data/financial_textual_dataset/balanced/risk/dataset-risk-eval.json'
TEST_DATASET='/media/SSD_2TB/financial_data/financial_textual_dataset/balanced/risk/dataset-risk-test.json'
FINANCIAL=true
LOAD_BEST_MODEL_AT_END=true

# Use --load_best_model_at_end setting or not
if [ "$LOAD_BEST_MODEL_AT_END" = true ]; then
  echo "Using --load_best_model_at_end setting."
  CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python tuning/ipo_longformer.py --model_name_or_path ${MODEL_NAME} --output_dir ipo-longformer-extra-global-txff-20480/risk/${TASK}/${MODEL_NAME} --do_train --do_eval --do_predict --overwrite_output_dir --evaluation_strategy epoch --save_strategy epoch --logging_strategy epoch --save_total_limit 5 --fp16 --fp16_full_eval --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS} --train_dataset_dir ${TRAIN_DATASET} --eval_dataset_dir ${EVAL_DATASET} --test_dataset_dir ${TEST_DATASET} --concatenate_financial_features ${FINANCIAL} --max_seq_length 20480 --max_segments 160 --max_seg_length 128 --load_best_model_at_end true --metric_for_best_model macro-avg-auc --greater_is_better true
else
  echo "Not using --load_best_model_at_end setting."
  CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python tuning/ipo_longformer.py --model_name_or_path ${MODEL_NAME} --output_dir ipo-longformer-extra-global-txff-20480/risk/${TASK}/${MODEL_NAME} --do_train --do_eval --do_predict --overwrite_output_dir --evaluation_strategy epoch --save_strategy epoch --logging_strategy epoch --save_total_limit 5 --fp16 --fp16_full_eval --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS} --train_dataset_dir ${TRAIN_DATASET} --eval_dataset_dir ${EVAL_DATASET} --test_dataset_dir ${TEST_DATASET} --concatenate_financial_features ${FINANCIAL} --max_seq_length 20480 --max_segments 160 --max_seg_length 128
fi