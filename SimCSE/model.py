import torch
import torch.nn as nn

from tnlrv3.modeling import TuringNLRv3ForSequenceClassification
from tnlrv3.configuration_tnlrv3 import TuringNLRv3Config


class AttentionPooling(nn.Module):
    def __init__(self, emb_size, hidden_size):
        super(AttentionPooling, self).__init__()
        self.att_fc1 = nn.Linear(emb_size, hidden_size)
        self.att_fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x, attn_mask=None):
        """
        Args:
            x: batch_size, candidate_size, emb_dim
            attn_mask: batch_size, candidate_size
        Returns:
            (shape) batch_size, emb_dim
        """
        e = self.att_fc1(x)
        e = nn.Tanh()(e)
        alpha = self.att_fc2(e)
        alpha = torch.exp(alpha)

        if attn_mask is not None:
            alpha = alpha * attn_mask.unsqueeze(2)

        alpha = alpha / (torch.sum(alpha, dim=1, keepdim=True) + 1e-8)
        x = torch.bmm(x.permute(0, 2, 1), alpha).squeeze(dim=-1)
        return x


class NewsEncoder(nn.Module):
    def __init__(self, args):
        super(NewsEncoder, self).__init__()
        self.bert_config = TuringNLRv3Config.from_pretrained(
            '../unilmv2/unilm2-base-uncased-config.json',
            output_hidden_states=False,
            num_hidden_layers=12)
        self.bert_model = TuringNLRv3ForSequenceClassification.from_pretrained(
            '../unilmv2/unilm2-base-uncased.bin', config=self.bert_config)
        self.attn = AttentionPooling(self.bert_config.hidden_size, args.news_query_vector_dim)
        self.dense = nn.Linear(self.bert_config.hidden_size, args.news_dim)

    def forward(self, text_ids):
        '''
            text_ids: batch_size, word_num
        '''
        text_attmask = text_ids.ne(0).float()
        word_vecs = self.bert_model(text_ids, text_attmask)[0]
        news_vec = self.attn(word_vecs)
        news_vec = self.dense(news_vec)
        return news_vec


class SimCSEModel(nn.Module):
    def __init__(self, args):
        super(SimCSEModel, self).__init__()
        self.temp = args.temp
        self.news_encoder = NewsEncoder(args)
        self.loss = nn.CrossEntropyLoss()
        self.label = torch.arange(args.batch_size).cuda()
        self.cosine_similarity = nn.CosineSimilarity(dim=-1)

    def forward(self, body):
        '''
            body: bz, MAX_BODY_WORD
        '''
        bz, _ = body.shape
        double_batch = torch.cat([body, body], dim=0)
        double_batch_emb = self.news_encoder(double_batch)
        batch_one_emb = torch.narrow(double_batch_emb, 0, 0, bz)
        batch_two_emb = torch.narrow(double_batch_emb, 0, bz, bz)

        cos_sim = self.cosine_similarity(batch_one_emb.unsqueeze(1), batch_two_emb.unsqueeze(0)) / self.temp
        label = torch.narrow(self.label, 0, 0, bz)
        loss = self.loss(cos_sim, label)
        return loss
