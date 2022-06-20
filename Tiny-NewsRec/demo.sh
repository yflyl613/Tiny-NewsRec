#!/bin/bash

hvd_size=4

mode=$1
exp_name='Tiny-NewsRec-4'
model_dir=../model_all/${exp_name}
batch_size=32
npratio=4
epochs=4
lr=0.0001
num_words_title=30
word_embedding_dim=768
freeze_embedding=False
num_attention_heads=16
news_dim=256
save_steps=50000
max_steps_per_epoch=7000
apply_bert=True
num_teacher_layers=12
num_student_layers=4
bert_trainable_layer="2 3"
model='NAML'
model_type='tnlrv3'
model_name=../unilmv2/unilm2-base-uncased.bin
config_name=../unilmv2/unilm2-base-uncased-config.json
tokenizer_name=../unilmv2/unilm2-base-uncased-vocab.txt
pooling='att'
temperature=1.0
coef=0.2
teacher_ckpts='../PLM-NR-12(DP)-1.pt ../PLM-NR-12(DP)-2.pt ../PLM-NR-12(DP)-3.pt ../PLM-NR-12(DP)-4.pt'
teacher_emb_paths='../PLM-NR-12(DP)-1.pkl ../PLM-NR-12(DP)-2.pkl ../PLM-NR-12(DP)-3.pkl ../PLM-NR-12(DP)-4.pkl'
num_teachers=4

train_data_dir=../MIND/MINDlarge_train
test_data_dir=../MIND/MINDlarge_train

if [ ${mode} == train ]
then
 user_log_mask=False
 filename_pat='behaviors_np4_*.tsv'
 use_pretrain_model=True
 pretrain_model_path='../first_stage_4_layer.pt'
 mpirun -np ${hvd_size} -H localhost:${hvd_size} \
 python -u run.py --mode ${mode} --model_dir ${model_dir} --batch_size ${batch_size} --npratio ${npratio} --temperature ${temperature} --coef ${coef} \
 --train_data_dir ${train_data_dir} --test_data_dir ${test_data_dir} \
 --epochs ${epochs} --lr ${lr} --num_words_title ${num_words_title} --word_embedding_dim ${word_embedding_dim} \
 --use_pretrain_model ${use_pretrain_model} --pretrain_model_path ${pretrain_model_path} \
 --teacher_ckpts ${teacher_ckpts} --teacher_emb_paths ${teacher_emb_paths} --num_teachers ${num_teachers} \
 --freeze_embedding ${freeze_embedding} --news_dim ${news_dim} --save_steps ${save_steps} --user_log_mask ${user_log_mask} \
 --max_steps_per_epoch ${max_steps_per_epoch} --apply_bert ${apply_bert} --filename_pat ${filename_pat} --num_attention_heads ${num_attention_heads} \
 --num_teacher_layers ${num_teacher_layers} --num_student_layers ${num_student_layers} --bert_trainable_layer ${bert_trainable_layer} --model ${model} --model_type ${model_type} \
 --model_name ${model_name} --config_name ${config_name} --tokenizer_name ${tokenizer_name} --pooling ${pooling} | tee ../log_all/${exp_name}_train.txt
elif [ ${mode} == test ]
then
 user_log_mask=True
 batch_size=128
 load_ckpt_name=$2
 filename_pat='behaviors_*.tsv'
 python -u run.py --mode ${mode} --model_dir ${model_dir} --batch_size ${batch_size} --npratio ${npratio} --filename_pat ${filename_pat} \
 --train_data_dir ${train_data_dir} --test_data_dir ${test_data_dir} \
 --epochs ${epochs} --lr ${lr} --num_words_title ${num_words_title} --word_embedding_dim ${word_embedding_dim} --num_teachers ${num_teachers} \
 --freeze_embedding ${freeze_embedding} --news_dim ${news_dim} --save_steps ${save_steps} --user_log_mask ${user_log_mask} \
 --max_steps_per_epoch ${max_steps_per_epoch} --apply_bert ${apply_bert} --load_ckpt_name ${load_ckpt_name} --num_attention_heads ${num_attention_heads} \
 --num_teacher_layers ${num_teacher_layers} --num_student_layers ${num_student_layers} --bert_trainable_layer ${bert_trainable_layer} --model ${model} --model_type ${model_type} \
 --model_name ${model_name} --config_name ${config_name} --tokenizer_name ${tokenizer_name} --pooling ${pooling} | tee ../log_all/${exp_name}_${load_ckpt_name}_test.txt
elif [ ${mode} == get_teacher_emb ]
then
 user_log_mask=False
 python -u run.py --mode ${mode} --model_dir ${model_dir} --batch_size ${batch_size} \
 --train_data_dir ${train_data_dir} --test_data_dir ${test_data_dir} \
 --num_words_title ${num_words_title} --word_embedding_dim ${word_embedding_dim} --teacher_ckpts ${teacher_ckpts} \
 --news_dim ${news_dim} --apply_bert ${apply_bert} --num_attention_heads ${num_attention_heads} --teacher_emb_paths ${teacher_emb_paths} \
 --num_hidden_layers 12 --model ${model} --model_type ${model_type} \
 --model_name ${model_name} --config_name ${config_name} --tokenizer_name ${tokenizer_name} --pooling ${pooling}
else
 echo "please enter a train or test"
fi