## Tiny-NewsRec

The source codes for our paper "Tiny-NewsRec: Efﬁcient and Effective PLM-based News Recommendation".

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

  First, you need to run the notebook `Further_Pre-train.ipynb` to further pre-train the 12-layer UniLMv2 with the MLM task. This will generate a checkpoint named `FP_12_layer.pt` under `Tiny-NewsRec/`.

  Then you can use the script `Tiny-NewsRec/PLM-NR/demo.sh` to finetune it with the news recommendation task. Remember to set `use_pretrain_model` as `True` and set `pretrain_model_path` as `../FP_12_layer.pt`.

- **PLM-NR (DP)**

  First, you need to run the notebook `Domain-specific_Post-train.ipynb` to domain-specifically post-train the 12-layer UniLMv2. This will generate a checkpoint named `DP_12_layer.pt` under `Tiny-NewsRec/`. It will also generate two `.pkl` files named `teacher_title_emb.pkl` and `teacher_body_emb.pkl` which are used for the first stage knowledge distillation in our Tiny-NewsRec method.

  Then you can use the script `Tiny-NewsRec/PLM-NR/demo.sh` to finetune it with the news recommendation task. Remembert to set `use_pretrain_model` as `True` and set `pretrain_model_path` as `../DP_12_layer.pt`.

- **TinyBERT**

  `Tiny-NewsRec/TinyBERT/demo.sh` is the script used to train TinyBERT.

  Set `hvd_size` to the number of available GPUs. Modify the value of `num_student_layers` to change the number of Transformer layers in the student model and set `bert_trainable_layers` to the indexes of its last two layers (start from 0). Set `teacher_ckpt` as the path to the previous PLM-NR-12 (DP) checkpoint. Set `use_pretrain_model` as `False` and then you can start training with `bash demo.sh train`.

- **NewsBERT**

  `Tiny-NewsRec/NewsBERT/demo.sh` is the script used to train NewsBERT.

  Set `hvd_size` to the number of available GPUs. Modify the value of `num_student_layers` to change the number of Transformer layers in the student model and set `student_trainable_layers` to the indexes of its last two layers (start from 0). Set `teacher_ckpt` as `../DP_12_layer.pt` to initialize the teacher model with the domain-specifically post-trained UniLMv2 and then you can start training with `bash demo.sh train`.

- **Tiny-NewsRec**

  First, you need to train 4 PLM-NR-12 (DP) as the teacher models.

  Second, you need to run the notebook `First-Stage.ipynb` to run the first-stage knowledge distillation in our approach. Modify `args.num_hidden_layers` to change the number of Transformer layers in the student model. This will generate a checkpoint of the student model under `Tiny-NewsRec/`.

  Then you need to run `bash demo.sh get_teacher_emb` under `Tiny-NewsRec/Tiny-NewsRec` to generate the news embeddings of the teacher models. Set `teacher_ckpts` as the path to the teacher models (separate by space).

  Finally, you can run the second-stage knowledge distillation in our approach with the script `Tiny-NewsRec/Tiny-NewsRec/demo.sh`. Modify the value of `num_student_layers` to change the number of Transformer layers in the student model and set `bert_trainable_layers` to the indexes of its last two layers (start from 0). Set `use_pretrain_model` as `True` and set `pretrain_model_path` as the path to the checkpoint generated by the notebook  `First-Stage.ipynb`. Then you can start training with `bash demo.sh train`.

### Citation

If you want to cite Tiny-NewsRec in your papers, you can cite it as follows:

```
@article{yu2021tinynewsrec,
    title={Tiny-NewsRec: Efficient and Effective PLM-based News Recommendation},
    author={Yang Yu and Fangzhao Wu and Chuhan Wu and Jingwei Yi and Tao Qi and Qi Liu},
    year={2021},
    journal={arXiv preprint arXiv:2112.00944}
}
```
