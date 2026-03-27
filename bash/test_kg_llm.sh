#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1
export TORCH_DISTRIBUTED_DEBUG=DETAIL


python aligner/tune/train_kg_llm.py \
        --freeze_llama \
        --inference \
        --best_epoch 0 \
        --single_modal \
        --dataset $1 \
        --test_dataset $2 \
        --att_d_model 2560 \
        --gnn_output 4096 \
	    --grad_steps 1 \
        --batch_size 3 \
        --num_token $3 \
        --clip_grad_norm 1.0 \
        --backbone $6 \
        --epoch 1 \
	    --weight_decay 0.1 \
        --max_text_length $4 \
        --prefix $5 \
        --pretrain_gnn $7\
        --instruction_path $8\
        --kg_pt_path $9\
        --gnn_model_path ${10}\
        --dataset_path ${11}\
        --result_save_path ${12}\
        --data_num ${13}\
