#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1
export TORCH_DISTRIBUTED_DEBUG=DETAIL
#swanlab offline

dataset="icews_wiki"
dataset_path="data/"
data_num="all"
result_save_path="./results/icews_wiki/result_all.txt"
num_epochs=1
num_hidden=2560
num_out=4096
batch_size=256
gnn_type='sage'
experiment_name='transe'
prefix='aligner'
pretrain_gnn='GNN'
transe_epochs=60

#训练transe
python mixer/main.py \
       --experiment_name $experiment_name \
       --hidden_dim $num_hidden \
       --dataset $dataset \
       --batch_size $batch_size \
       --transe_epochs $transe_epochs \


python aligner/pretrain/pretrain_EA.py \
        --dataset $dataset \
        --dataset_path $dataset_path \
        --result_save_path $result_save_path\
        --data_num $data_num\
        --single_modal \
        --num_epochs $num_epochs \
        --num_hidden $num_hidden \
	      --num_out $num_out \
        --batch_size $batch_size \
        --gnn_type $gnn_type \
        --pretrain_gnn $pretrain_gnn \


datasets=('icews_wiki:2000')


instruction_path="data/instruction/"
kg_pt_path="data/kg/"
gnn_model_path="saved_model/gnn/"


num_token=5


pred_file="./results/icews_wiki/aligner_icews_wiki_model_resultsall.txt"
label_file="./results/icews_wiki/aligner_icews_wiki_model_labelsall.txt"

llm='vicuna-7b-v1___5'


python aligner/tune/train_kg_llm.py \
        --dataset $dataset \
        --test_dataset $dataset \
        --result_save_path $result_save_path\
        --data_num $data_num\
        --single_modal \
        --pretrain_gnn $pretrain_gnn\
        --dataset_path $dataset_path \
        --instruction_path $instruction_path\
        --kg_pt_path $kg_pt_path\
        --gnn_model_path $gnn_model_path\
        --att_d_model 2560 \
        --gnn_output 4096 \
	      --grad_steps 1 \
        --batch_size 1 \
        --max_text_length 760\
        --num_token $num_token \
        --clip_grad_norm 1.0 \
        --backbone $llm \
        --epoch 1 \
	      --weight_decay 0.1 \
        --prefix $prefix \



for pair in "${datasets[@]}"
do
    IFS=':' read -r test_dataset max_text_length <<< "$pair"

    echo "Testing on KG $test_dataset"
    bash ./bash/test_kg_llm.sh $dataset $test_dataset $num_token $max_text_length $prefix $llm $pretrain_gnn $instruction_path $kg_pt_path $gnn_model_path $dataset_path $result_save_path $data_num
done

python aligner/tune/evaluate_ea_predictions.py \
      --dataset $dataset\
      --result_save_path $result_save_path\
      --pred_file $pred_file\
      --label_file $label_file\
      --verbose
