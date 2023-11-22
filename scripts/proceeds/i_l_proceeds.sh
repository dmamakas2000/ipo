GPU_NUMBER=0
MODEL_NAME='ipo-longformer'
BATCH_SIZE=1
ACCUMULATION_STEPS=1
TASK='IPO'
TRAIN_DATASET='/media/SSD_2TB/financial_data/financial_textual_dataset/balanced/proceeds/dataset-proceeds-train.json'
EVAL_DATASET='/media/SSD_2TB/financial_data/financial_textual_dataset/balanced/proceeds/dataset-proceeds-eval.json'
TEST_DATASET='/media/SSD_2TB/financial_data/financial_textual_dataset/balanced/proceeds/dataset-proceeds-test.json'
LEARNING_RATE=7.89819931348775e-05
EPOCHS=10
MAX_SEGMENTS=160
MAX_SEGMENT_LENGTH=128
MAX_SEQUENCE_LENGTH=$((MAX_SEGMENTS * MAX_SEGMENT_LENGTH))
FINANCIAL=true
REDUCTION_FEATURES=14
LOAD_BEST_MODEL_AT_END=true
THRESHOLD=0.5292290045898985

# Run the seeds
for i in {1..5}; do
  # Choose proper way to log
  if [ "$FINANCIAL" = true ]; then
    MODEL_TYPE='txff'
  else
    MODEL_TYPE='tx'
  fi

  # Use --load_best_model_at_end setting or not
  if [ "$LOAD_BEST_MODEL_AT_END" = true ]; then
    CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python experiments/ipo_longformer.py --model_name_or_path ${MODEL_NAME} --output_dir ${MODEL_NAME}-${MODEL_TYPE}-cls-${MAX_SEQUENCE_LENGTH}/${TASK}/proceeds/${MODEL_NAME}/seed_${i} --do_train --do_eval --do_pred --overwrite_output_dir --evaluation_strategy epoch --save_strategy epoch --save_total_limit 5 --num_train_epochs ${EPOCHS} --learning_rate ${LEARNING_RATE} --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --seed ${i} --fp16 --fp16_full_eval --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS} --train_dataset_dir ${TRAIN_DATASET} --eval_dataset_dir ${EVAL_DATASET} --test_dataset_dir ${TEST_DATASET} --max_seq_length ${MAX_SEQUENCE_LENGTH} --max_segments ${MAX_SEGMENTS} --max_seg_length ${MAX_SEGMENT_LENGTH} --concatenate_financial_features ${FINANCIAL} --reduction_features ${REDUCTION_FEATURES} --threshold ${THRESHOLD} --load_best_model_at_end true --metric_for_best_model macro-avg-auc --greater_is_better true
  else
    CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python experiments/ipo_longformer.py --model_name_or_path ${MODEL_NAME} --output_dir ${MODEL_NAME}-${MODEL_TYPE}-cls-${MAX_SEQUENCE_LENGTH}/${TASK}/proceeds/${MODEL_NAME}/seed_${i} --do_train --do_eval --do_pred --overwrite_output_dir --evaluation_strategy epoch --save_strategy epoch --save_total_limit 5 --num_train_epochs ${EPOCHS} --learning_rate ${LEARNING_RATE} --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --seed ${i} --fp16 --fp16_full_eval --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS} --train_dataset_dir ${TRAIN_DATASET} --eval_dataset_dir ${EVAL_DATASET} --test_dataset_dir ${TEST_DATASET} --max_seq_length ${MAX_SEQUENCE_LENGTH} --max_segments ${MAX_SEGMENTS} --max_seg_length ${MAX_SEGMENT_LENGTH} --concatenate_financial_features ${FINANCIAL} --reduction_features ${REDUCTION_FEATURES} --threshold ${THRESHOLD}
  fi
done