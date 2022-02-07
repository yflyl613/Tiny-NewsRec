import numpy as np
import torch
import logging
from tqdm.auto import tqdm
import torch.optim as optim
import utils
import os
from pathlib import Path
import random
from dataloader import DataLoaderTrain, DataLoaderTest
from torch.utils.data import Dataset, DataLoader
from streaming import get_stat, get_worker_files

from parameters import parse_args
from preprocess import read_news_bert, get_doc_input_bert
from model_bert import ModelBert


def train(args):
    if args.enable_hvd:
        import horovod.torch as hvd

    if args.load_ckpt_name is not None:
        ckpt_path = utils.get_checkpoint(args.model_dir, args.load_ckpt_name)
    else:
        ckpt_path = utils.latest_checkpoint(args.model_dir)

    hvd_size, hvd_rank, hvd_local_rank = utils.init_hvd_cuda(
        args.enable_hvd, args.enable_gpu)

    stat = get_stat(args.train_data_dir, args.filename_pat)
    print(stat)

    data_paths = get_worker_files(args.train_data_dir,
                                  hvd_rank, hvd_size, args.filename_pat, args.enable_shuffle, 0
                                  )

    sample_num = 0
    for file in data_paths:
        sample_num += stat[file]

    logging.info("[{}] contains {} samples {} steps".format(
        hvd_rank, sample_num, sample_num // args.batch_size))

    news, news_index, category_dict, subcategory_dict = read_news_bert(
        os.path.join(args.train_data_dir, 'news.tsv'), args, mode='train'
    )

    news_title, news_title_attmask, news_category, news_subcategory = get_doc_input_bert(
        news, news_index, category_dict, subcategory_dict, args)

    news_combined = np.concatenate([news_title, news_title_attmask], axis=-1)

    model = ModelBert(args)

    if args.use_pretrain_model:
        ckpt = torch.load(args.pretrain_model_path, map_location='cpu')
        pretrained_dict = ckpt["model_state_dict"]
        model_dict = model.state_dict()
        remain_key = list(model_dict.keys())
        pretrained_key = []
        for k, v in pretrained_dict.items():
            if not k.startswith('student'):
                continue
            key = k
            model_dict[key].copy_(v)
            pretrained_key.append(key)
            remain_key.remove(key)

        model.load_state_dict(model_dict)

        if hvd_rank == 0:
            logging.info(f"loaded pretrain model: {args.pretrain_model_path}")
            print(f'{len(pretrained_key)} loaded pretrained parameters:')
            for k in pretrained_key:
                print(f'\t{k}')
            print(f'{len(remain_key)} randomly initialized parameters:')
            for k in remain_key:
                print(f'\t{k}')

        del ckpt
        torch.cuda.empty_cache()

        for param in model.news_encoder.bert_model.parameters():
            param.requires_grad = False

        for index, layer in enumerate(model.news_encoder.bert_model.bert.encoder.layer):
            if index in args.bert_trainable_layer:
                logging.info(f"finetune block {index}")
                for param in layer.parameters():
                    param.requires_grad = True

        if args.enable_gpu:
            model = model.cuda()

        pretrained_param = []
        rest_param = []
        for name, param in model.named_parameters():
            if name in pretrained_key:
                pretrained_param.append(param)
            else:
                rest_param.append(param)

        optimizer = torch.optim.Adam([
            {'params': pretrained_param, 'lr': args.pretrain_lr},
            {'params': rest_param, 'lr': args.lr}], amsgrad=True)

    else:
        if args.model_type == 'tnlrv3':
            for param in model.news_encoder.bert_model.parameters():
                param.requires_grad = False

            for index, layer in enumerate(model.news_encoder.bert_model.bert.encoder.layer):
                if index in args.bert_trainable_layer:
                    logging.info(f"finetune block {index}")
                    for param in layer.parameters():
                        param.requires_grad = True
        else:
            for param in model.news_encoder.bert_model.parameters():
                param.requires_grad = False

            for index, layer in enumerate(model.news_encoder.bert_model.encoder.layer):
                if index in args.bert_trainable_layer:
                    logging.info(f"finetune block {index}")
                    for param in layer.parameters():
                        param.requires_grad = True

        if args.enable_gpu:
            model = model.cuda()

        optimizer = optim.Adam(model.parameters(), lr=args.lr, amsgrad=True)

    word_dict = None

    if args.load_ckpt_name is not None:
        ckpt_path = utils.get_checkpoint(args.model_dir, args.load_ckpt_name)
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        logging.info(f"Model loaded from {ckpt_path}")

    if hvd_rank == 0:
        print(model)
        for name, param in model.named_parameters():
            print(name, param.requires_grad)

    if args.enable_hvd:
        hvd.broadcast_parameters(model.state_dict(), root_rank=0)
        hvd.broadcast_optimizer_state(optimizer, root_rank=0)
        compression = hvd.Compression.none
        optimizer = hvd.DistributedOptimizer(
            optimizer,
            named_parameters=model.named_parameters(),
            compression=compression,
            op=hvd.Average)

    dataloader = DataLoaderTrain(
        news_index=news_index,
        news_combined=news_combined,
        word_dict=word_dict,
        data_dir=args.train_data_dir,
        filename_pat=args.filename_pat,
        args=args,
        world_size=hvd_size,
        worker_rank=hvd_rank,
        cuda_device_idx=hvd_local_rank,
        enable_prefetch=True,
        enable_shuffle=True,
        enable_gpu=args.enable_gpu,
    )

    logging.info('Training...')
    for ep in range(args.start_epoch, args.epochs):
        loss = 0.0
        accuary = 0.0
        for cnt, (log_ids, log_mask, input_ids, targets) in enumerate(dataloader):
            if cnt > args.max_steps_per_epoch:
                break

            bz_loss, y_hat = model(log_ids, log_mask, input_ids, targets)
            loss += bz_loss.data.float()
            accuary += utils.acc(targets, y_hat)
            optimizer.zero_grad()
            bz_loss.backward()
            optimizer.step()

            if cnt % args.log_steps == 0:
                logging.info(
                    '[{}] Ed: {}, train_loss: {:.5f}, acc: {:.5f}'.format(
                        hvd_rank, cnt * args.batch_size, loss.data / cnt,
                        accuary / cnt))

        loss /= cnt
        print(ep + 1, loss)

        # save model last of epoch
        if hvd_rank == 0:
            ckpt_path = os.path.join(args.model_dir, f'epoch-{ep+1}.pt')
            torch.save(
                {
                    'model_state_dict': model.state_dict(),
                    'category_dict': category_dict,
                    'word_dict': word_dict,
                    'subcategory_dict': subcategory_dict
                }, ckpt_path)
            logging.info(f"Model saved to {ckpt_path}")

    dataloader.join()


def test(args):
    if args.enable_hvd:
        import horovod.torch as hvd

    hvd_size, hvd_rank, hvd_local_rank = utils.init_hvd_cuda(
        args.enable_hvd, args.enable_gpu)

    if args.load_ckpt_name is not None:
        ckpt_path = utils.get_checkpoint(args.model_dir, args.load_ckpt_name)
    else:
        ckpt_path = utils.latest_checkpoint(args.model_dir)

    assert ckpt_path is not None, 'No ckpt found'
    checkpoint = torch.load(ckpt_path)

    subcategory_dict = checkpoint['subcategory_dict']
    category_dict = checkpoint['category_dict']
    word_dict = checkpoint['word_dict']

    model = ModelBert(args)

    if args.enable_gpu:
        model.cuda()

    model.load_state_dict(checkpoint['model_state_dict'])
    logging.info(f"Model loaded from {ckpt_path}")

    if args.enable_hvd:
        hvd.broadcast_parameters(model.state_dict(), root_rank=0)

    model.eval()
    torch.set_grad_enabled(False)

    news, news_index = read_news_bert(
        os.path.join(args.test_data_dir, 'news.tsv'), args, mode='test'
    )

    news_title, news_title_attmask, news_category, news_subcategory = get_doc_input_bert(
        news, news_index, category_dict, subcategory_dict, args)

    news_combined = np.concatenate([news_title, news_title_attmask], axis=1)

    class NewsDataset(Dataset):
        def __init__(self, data):
            self.data = data

        def __getitem__(self, idx):
            return self.data[idx]

        def __len__(self):
            return self.data.shape[0]

    def news_collate_fn(arr):
        arr = torch.LongTensor(arr)
        return arr

    news_dataset = NewsDataset(news_combined)
    news_dataloader = DataLoader(news_dataset,
                                 batch_size=args.batch_size * 4,
                                 num_workers=args.num_workers,
                                 collate_fn=news_collate_fn)

    news_scoring = []
    with torch.no_grad():
        for input_ids in tqdm(news_dataloader):
            input_ids = input_ids.cuda()
            news_vec = model.news_encoder(input_ids)
            news_vec = news_vec.to(torch.device("cpu")).detach().numpy()
            news_scoring.extend(news_vec)

    news_scoring = np.array(news_scoring)
    logging.info("news scoring num: {}".format(news_scoring.shape[0]))

    doc_sim = 0
    for _ in tqdm(range(1000000)):
        i = random.randrange(1, len(news_scoring))
        j = random.randrange(1, len(news_scoring))
        if i != j:
            doc_sim += np.dot(news_scoring[i], news_scoring[j]) / (
                np.linalg.norm(news_scoring[i]) * np.linalg.norm(news_scoring[j]))
    print(f'=== doc-sim: {doc_sim / 1000000} ===')

    dataloader = DataLoaderTest(
        news_index=news_index,
        news_scoring=news_scoring,
        word_dict=word_dict,
        data_dir=args.test_data_dir,
        filename_pat=args.filename_pat,
        args=args,
        world_size=hvd_size,
        worker_rank=hvd_rank,
        cuda_device_idx=hvd_local_rank,
        enable_prefetch=True,
        enable_shuffle=False,
        enable_gpu=args.enable_gpu,
    )

    from metrics import roc_auc_score, ndcg_score, mrr_score

    AUC = []
    MRR = []
    nDCG5 = []
    nDCG10 = []

    def print_metrics(hvd_local_rank, cnt, x):
        logging.info("[{}] Ed: {}: {}".format(hvd_local_rank, cnt,
                                              '\t'.join(["{:0.2f}".format(i * 100) for i in x])))

    def get_mean(arr):
        return [np.array(i).mean() for i in arr]

    def get_sum(arr):
        return [np.array(i).sum() for i in arr]

    local_sample_num = 0

    for cnt, (log_vecs, log_mask, news_vecs, labels) in enumerate(dataloader):

        local_sample_num += log_vecs.shape[0]

        user_vecs = model.user_encoder(log_vecs, log_mask).to(
            torch.device("cpu")).detach().numpy()

        for user_vec, news_vec, label in zip(user_vecs, news_vecs, labels):

            if label.mean() == 0 or label.mean() == 1:
                continue

            score = np.dot(news_vec, user_vec)

            auc = roc_auc_score(label, score)
            mrr = mrr_score(label, score)
            ndcg5 = ndcg_score(label, score, k=5)
            ndcg10 = ndcg_score(label, score, k=10)

            AUC.append(auc)
            MRR.append(mrr)
            nDCG5.append(ndcg5)
            nDCG10.append(ndcg10)

        if cnt % args.log_steps == 0:
            print_metrics(hvd_rank, local_sample_num,
                          get_mean([AUC, MRR, nDCG5, nDCG10]))

    # stop scoring
    dataloader.join()

    logging.info('[{}] local_sample_num: {}'.format(
        hvd_rank, local_sample_num))
    total_sample_num = hvd.allreduce(
        torch.tensor(local_sample_num), op=hvd.Sum)
    local_metrics_sum = get_sum([AUC, MRR, nDCG5, nDCG10])
    total_metrics_sum = hvd.allreduce(torch.tensor(
        local_metrics_sum, dtype=float), op=hvd.Sum)
    if hvd_rank == 0:
        print_metrics(hvd_rank, total_sample_num,
                      total_metrics_sum / total_sample_num)


if __name__ == "__main__":
    utils.setuplogger()
    args = parse_args()
    Path(args.model_dir).mkdir(parents=True, exist_ok=True)
    if 'train' in args.mode:
        train(args)
    if 'test' in args.mode:
        test(args)
