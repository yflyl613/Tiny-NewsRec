## Tiny-NewsRec

The source code and data for our paper "Tiny-NewsRec: Effective and Efficient PLM-based News Recommendation" in EMNLP 2022.

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

Then, you should run `python split_file.py` under `Tiny-NewsRec/` to prepare the training data. Set `N` in line 13 of `split_file.py` as the number of available GPUs. This script will construct the training samples and split them into `N` files for multi-GPU training.

### Experiments

Our Tiny-NewsRec method contains the following 4 steps:

- **Step 1**

  Run the notebook `Domain-specific_Post-train.ipynb` to domain-specifically post-train the PLM-based news encoder. This will generate a checkpoint named `DP_12_layer_{step}.pt` every `args.T` steps under `Tiny-NewsRec/`. Then you need to set the variable `ckpt_paths` as the paths to the last $M$ checkpoints and run the rest cells. For each checkpoint, it will generate two `.pkl` files named `teacher_title_emb_{idx}.pkl` and `teacher_body_emb_{idx}.pkl` which are used for the post-training stage knowledge distillation in our method.

- **Step 2**

  Run the notebook `Post-train_KD.ipynb` to run the post-training stage knowledge distillation in our method. Modify `args.num_hidden_layers` to change the number of Transformer layers in the student model. This will generate a checkpoint of the student model under `Tiny-NewsRec/`.

- **Step 3**

  Use the script `Tiny-NewsRec/PLM-NR/demo.sh` to finetune the $M$ teacher models post-trained in Step 1 with the news recommendation task. Remember to set `use_pretrain_model` as `True` and set `pretrain_model_path` as the path to one of these teacher models respectively.

- **Step 4**

  Run `bash demo.sh get_teacher_emb` under `Tiny-NewsRec/Tiny-NewsRec` to generate the news embeddings of the $M$ teacher model finetuned in Step 3 for the finetuning stage knowledge distillation in our method. Set `teacher_ckpts` as the path to these teacher models (separate by space).

  Use the script `Tiny-NewsRec/Tiny-NewsRec/demo.sh` to run the finetuning stage knowledge distillation in our method. Modify the value of `num_student_layers` to change the number of Transformer layers in the student model and set `bert_trainable_layers` to the indexes of its last two layers (start from 0). Set `use_pretrain_model` as `True` and set `pretrain_model_path` as the path to the checkpoint of the student model generated in Step 2. Then you can start training with `bash demo.sh train`.
