# Experiment's settings
GPU_NUMBER=4
MODEL_NAME='nlpaueb/sec-bert-base'
BATCH_SIZE=25
ACCUMULATION_STEPS=1
TASK='IPO'
TRAIN_DATASET='/media/SSD_2TB/financial_data/financial_textual_dataset/balanced/summary/dataset-summary-train.json'
EVAL_DATASET='/media/SSD_2TB/financial_data/financial_textual_dataset/balanced/summary/dataset-summary-eval.json'
TEST_DATASET='/media/SSD_2TB/financial_data/financial_textual_dataset/balanced/summary/dataset-summary-test.json'
LEARNING_RATE=1e-5
EPOCHS=10
MAX_POOLED=false
FINANCIAL=false
EMBEDDING_REDUCTION_FEATURES=8
LOAD_BEST_MODEL_AT_END=true
THRESHOLD=0.5

# Run the seeds
for i in {1..5}; do
  # Choose proper way to log the model type
  if [ "$FINANCIAL" = true ]; then
    MODEL_TYPE='txff'
  else
    MODEL_TYPE='tx'
  fi

  # Choose proper way to log the classification type
  if [ "$MAX_POOLED" = true ]; then
    CLASSIFICATION_TYPE='max'
  else
    CLASSIFICATION_TYPE='cls'
  fi

  # Use --load_best_model_at_end setting or not
  if [ "$LOAD_BEST_MODEL_AT_END" = true ]; then
    echo "Using --load_best_model_at_end setting."
    CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python experiments/ipo.py --model_name_or_path ${MODEL_NAME} --do_lower_case ${LOWER_CASE} --output_dir bert-${MODEL_TYPE}-${CLASSIFICATION_TYPE}-512/${TASK}/summary/${MODEL_NAME}/seed_${i} --do_train --do_eval --do_predict --overwrite_output_dir --evaluation_strategy epoch --save_strategy epoch --logging_strategy epoch --save_total_limit 5 --num_train_epochs ${EPOCHS} --learning_rate ${LEARNING_RATE} --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --seed ${i} --fp16 --fp16_full_eval --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS} --train_dataset_dir ${TRAIN_DATASET} --eval_dataset_dir ${EVAL_DATASET} --test_dataset_dir ${TEST_DATASET} --max_pooled ${MAX_POOLED} --hierarchical false --reduction_features ${EMBEDDING_REDUCTION_FEATURES} --concatenate_financial_features ${FINANCIAL} --threshold ${THRESHOLD} --load_best_model_at_end true --metric_for_best_model macro-avg-auc --greater_is_better true
  else
    echo "Not using --load_best_model_at_end setting."
    CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python experiments/ipo.py --model_name_or_path ${MODEL_NAME} --do_lower_case ${LOWER_CASE} --output_dir bert-${MODEL_TYPE}-${CLASSIFICATION_TYPE}-512/${TASK}/summary/${MODEL_NAME}/seed_${i} --do_train --do_eval --do_predict --overwrite_output_dir --evaluation_strategy epoch --save_strategy epoch --logging_strategy epoch --save_total_limit 5 --num_train_epochs ${EPOCHS} --learning_rate ${LEARNING_RATE} --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --seed ${i} --fp16 --fp16_full_eval --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS} --train_dataset_dir ${TRAIN_DATASET} --eval_dataset_dir ${EVAL_DATASET} --test_dataset_dir ${TEST_DATASET} --max_pooled ${MAX_POOLED} --hierarchical false --reduction_features ${EMBEDDING_REDUCTION_FEATURES} --concatenate_financial_features ${FINANCIAL} --threshold ${THRESHOLD}
  fi
done
