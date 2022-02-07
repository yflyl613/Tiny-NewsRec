import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from tqdm.auto import tqdm
import logging
import sys

from parameters import parse_args
from model import SimCSEModel
from tnlrv3.tokenization_tnlrv3 import TuringNLRv3Tokenizer

root = logging.getLogger()
root.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
formatter = logging.Formatter("[%(levelname)s %(asctime)s] %(message)s")
handler.setFormatter(formatter)
root.addHandler(handler)

args = parse_args()

tokenizer = TuringNLRv3Tokenizer.from_pretrained('../unilmv2/unilm2-base-uncased-vocab.txt', do_lower_case=True)

file = '../docs_filter.tsv'
with open(file, encoding='utf-8') as f:
    total_lines = f.readlines()

corpus = []
for line in total_lines:
    splited = line.strip('\n').split('\t')
    nid, cate, subcate, title, body, abstract, url, time = splited
    corpus.append(body)


class PretrainDataset(Dataset):
    def __init__(self, corpus):
        self.corpus = corpus
        self.len = len(corpus)

    def __getitem__(self, idx):
        text = self.corpus[idx]
        tokenized_text = tokenizer(text, max_length=args.max_len, pad_to_max_length=True, truncation=True)
        input_ids = np.array(tokenized_text['input_ids'])
        return input_ids

    def __len__(self):
        return self.len


pretrain_ds = PretrainDataset(corpus)
pretrain_dl = DataLoader(pretrain_ds, batch_size=args.batch_size, num_workers=4, shuffle=True, pin_memory=True)

pretrain_model = SimCSEModel(args).cuda()

for param in pretrain_model.news_encoder.bert_model.parameters():
    param.requires_grad = False

for index, layer in enumerate(pretrain_model.news_encoder.bert_model.encoder.layer):
    if index in [9, 10, 11]:
        for param in layer.parameters():
            param.requires_grad = True

for name, p in pretrain_model.named_parameters():
    print(name, p.requires_grad)

rest_param = filter(
    lambda x: id(x) not in list(map(id, pretrain_model.news_encoder.bert_model.parameters())), pretrain_model.parameters())

optimizer = optim.Adam(
    [{'params': pretrain_model.news_encoder.bert_model.parameters(), 'lr': 1e-6},
     {'params': rest_param, 'lr': 1e-5}]
)

for ep in range(1):
    loss = 0.0
    cnt = 0
    pretrain_model.train()
    for input_ids in tqdm(pretrain_dl):
        input_ids = input_ids.cuda(non_blocking=True)

        bz_loss = pretrain_model(input_ids)
        loss += bz_loss.data.float()

        optimizer.zero_grad()
        bz_loss.backward()
        optimizer.step()

        if cnt % 100 == 0:
            logging.info('Sample: {}, train_loss: {:.5f}'.format(cnt * args.batch_size, loss.data / cnt))

        cnt += 1

ckpt_path = '../SimCSE_12_layer.pt'
torch.save({'model_state_dict': pretrain_model.state_dict()}, ckpt_path)
