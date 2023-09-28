lang=$1 

# optimizer
lr=5e-5
batch_size=6
beam_size=10
epochs=10

# model 
source_length=200
target_length=30

# data
data_dir=dataset/$lang/contextual_medits
train_file=$data_dir/train.jsonl
dev_file=$data_dir/valid.jsonl
test_file=$data_dir/test.jsonl


pretrained_model=Salesforce/codet5-base 

# ============ Step 1 Training ==============

function train_codet5 () {

output_dir=saved_model/codet5/${lang}/
mkdir -p $output_dir
echo $output_dir
echo "============TRAINING============"
 CUDA_VISIBLE_DEVICES=0 python run.py  --do_train --do_eval   --do_test --eval_frequency 100 \
  --run_codet5 \
  --model_name_or_path $pretrained_model \
  --train_filename $train_file \
  --dev_filename $dev_file \
  --test_filename ${test_file} \
  --output_dir $output_dir \
  --max_source_length $source_length \
  --max_target_length $target_length \
  --do_lower_case \
  --beam_size $beam_size --train_batch_size $batch_size \
  --eval_batch_size $batch_size --learning_rate $lr \
  --num_train_epochs $epochs --seed 0 2>&1|tee  $output_dir/train.log
}


# 
train_codet5