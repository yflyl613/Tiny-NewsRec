## Tiny-NewsRec

The source codes for our paper "Efﬁcient and Effective News Recommendation with Pre-trained Language Models".

### Requirements

- PyTorch == 1.6.0
- TensorFlow == 1.15.0
- horovod == 0.19.5
- transformers == 3.0.2

### Prepare Data

You can download and unzip the public *MIND* dataset with the following command:

```bash
# Under Tiny-NewsRec/
mkdir MIND && mkdir log_all && mkdir model_all
cd MIND
wget https://mind201910small.blob.core.windows.net/release/MINDlarge_train.zip
wget https://mind201910small.blob.core.windows.net/release/MINDlarge_dev.zip
wget https://mind201910small.blob.core.windows.net/release/MINDlarge_test.zip
unzip MINDlarge_train.zip -d MINDlarge_train
unzip MINDlarge_dev.zip -d MINDlarge_dev
unzip MINDlarge_test.zip -d MINDlarge_test
cd ../
```

Then, you should run `python split_file.py` under `Tiny-NewsRec/` to prepare the training data. Set `N` in line 13 of `split_file.py` to the number of available GPUs. This script will construct the training samples and split them into `N` files for multi-GPU training.

### Experiments

- **PLM-NR (FT)**

  `Tiny-NewsRec/PLM-NR/demo.sh` is the script used to train PLM-NR (FT).

  Set `hvd_size` to the number of available GPUs. Modify the value of `num_hidden_layers` to change the number of Transformer layers in the PLM and set `bert_trainable_layers` to the indexes of its last two layers (start from 0). Set `use_pretrain_model` as `False` and then you can start training with `bash demo.sh train`.

- **PLM-NR (FP)**

  First, you need to run the notebook `Further_Pre-train.ipynb` to further pre-train the PLM-based news encoder with the MLM task. This will generate a checkpoint named `FP_12_layer.pt` under `Tiny-NewsRec/`.

  Then you can use the script `Tiny-NewsRec/PLM-NR/demo.sh` to finetune it with the news recommendation task. Remember to set `use_pretrain_model` as `True` and set `pretrain_model_path` as `../FP_12_layer.pt`.

- **PLM-NR (SimCSE)**

  First, you need to run `Tiny-NewRec/SimCSE/demo.sh` to train the PLM-based news encoder with the unsupervised SimCSE method. This will generate a checkpoint named `SimCSE_12_layer.pt` under `Tiny-NewsRec/`.

  Then you can use the script `Tiny-NewsRec/PLM-NR/demo.sh` to finetune it with the news recommendation task. Remember to set `use_pretrain_model` as `True` and set `pretrain_model_path` as `../SimCSE_12_layer.pt`.

- **PLM-NR (DP)**

  First, you need to run the notebook `Domain-specific_Post-train.ipynb` to domain-specifically post-train the PLM-based news encoder. This will generate a checkpoint named `DP_12_layer.pt` under `Tiny-NewsRec/`. It will also generate two `.pkl` files named `teacher_title_emb.pkl` and `teacher_body_emb.pkl` which are used for the first stage knowledge distillation in our Tiny-NewsRec method.

  Then you can use the script `Tiny-NewsRec/PLM-NR/demo.sh` to finetune it with the news recommendation task. Remembert to set `use_pretrain_model` as `True` and set `pretrain_model_path` as `../DP_12_layer.pt`.

- **TinyBERT**

  `Tiny-NewsRec/TinyBERT/demo.sh` is the script used to train TinyBERT.

  Set `hvd_size` to the number of available GPUs. Modify the value of `num_student_layers` to change the number of Transformer layers in the student model and set `bert_trainable_layers` to the indexes of its last two layers (start from 0). Set `teacher_ckpt` as the path to the previous PLM-NR-12 (DP) checkpoint. Set `use_pretrain_model` as `False` and then you can start training with `bash demo.sh train`.

- **NewsBERT**

  `Tiny-NewsRec/NewsBERT/demo.sh` is the script used to train NewsBERT.

  Set `hvd_size` to the number of available GPUs. Modify the value of `num_student_layers` to change the number of Transformer layers in the student model and set `student_trainable_layers` to the indexes of its last two layers (start from 0). Set `teacher_ckpt` as `../DP_12_layer.pt` to initialize the teacher model with the domain-specifically post-trained PLM-based news encoder and then you can start training with `bash demo.sh train`.

- **Tiny-NewsRec**

  First, you need to train a PLM-NR-12 (DP) model as the teacher model.

  Second, you need to run the notebook `First-Stage.ipynb` to run the first-stage knowledge distillation in our approach. Modify `args.num_hidden_layers` to change the number of Transformer layers in the student model. This will generate a checkpoint of the student model under `Tiny-NewsRec/`.

  Then you need to run `bash demo.sh get_teacher_emb` under `Tiny-NewsRec/Tiny-NewsRec` to generate the news embeddings of the finetuned teacher model for the second stage knowledge distillation. Set `teacher_ckpts` as the path to the previous PLM-NR-12 (DP) checkpoint.

  Finally, you can run the second-stage knowledge distillation in our approach with the script `Tiny-NewsRec/Tiny-NewsRec/demo.sh`. Modify the value of `num_student_layers` to change the number of Transformer layers in the student model and set `bert_trainable_layers` to the indexes of its last two layers (start from 0). Set `use_pretrain_model` as `True` and set `pretrain_model_path` as the path to the checkpoint of the student model generated by the notebook  `First-Stage.ipynb`. Then you can start training with `bash demo.sh train`.

