import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from utils import MODEL_CLASSES


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
        bz = x.shape[0]
        e = self.att_fc1(x)
        e = nn.Tanh()(e)
        alpha = self.att_fc2(e)
        alpha = torch.exp(alpha)

        if attn_mask is not None:
            alpha = alpha * attn_mask.unsqueeze(2)

        alpha = alpha / (torch.sum(alpha, dim=1, keepdim=True) + 1e-8)
        x = torch.bmm(x.permute(0, 2, 1), alpha).squeeze(dim=-1)
        return x


class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k

    def forward(self, Q, K, V, attn_mask=None):
        '''
            Q: batch_size, n_head, candidate_num, d_k
            K: batch_size, n_head, candidate_num, d_k
            V: batch_size, n_head, candidate_num, d_v
            attn_mask: batch_size, n_head, candidate_num
            Return: batch_size, n_head, candidate_num, d_v
        '''
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k)
        scores = torch.exp(scores)

        if attn_mask is not None:
            scores = scores * attn_mask.unsqueeze(dim=-2)

        attn = scores / (torch.sum(scores, dim=-1, keepdim=True) + 1e-8)
        context = torch.matmul(attn, V)
        return context


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, d_k, d_v):
        super(MultiHeadSelfAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_k
        self.d_v = d_v

        self.W_Q = nn.Linear(d_model, d_k * n_heads)
        self.W_K = nn.Linear(d_model, d_k * n_heads)
        self.W_V = nn.Linear(d_model, d_v * n_heads)

        self.scaled_dot_product_attn = ScaledDotProductAttention(self.d_k)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1)

    def forward(self, Q, K, V, mask=None):
        '''
            Q: batch_size, candidate_num, d_model
            K: batch_size, candidate_num, d_model
            V: batch_size, candidate_num, d_model
            mask: batch_size, candidate_num
        '''
        batch_size = Q.shape[0]
        if mask is not None:
            mask = mask.unsqueeze(dim=1).expand(-1, self.n_heads, -1)

        q_s = self.W_Q(Q).view(batch_size, -1, self.n_heads,
                               self.d_k).transpose(1, 2)
        k_s = self.W_K(K).view(batch_size, -1, self.n_heads,
                               self.d_k).transpose(1, 2)
        v_s = self.W_V(V).view(batch_size, -1, self.n_heads,
                               self.d_v).transpose(1, 2)

        context = self.scaled_dot_product_attn(q_s, k_s, v_s, mask)
        output = context.transpose(1, 2).contiguous().view(
            batch_size, -1, self.n_heads * self.d_v)
        return output


class NewsEncoder(nn.Module):
    def __init__(self, args, is_teacher):
        super(NewsEncoder, self).__init__()
        self.pooling = args.pooling
        config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
        self.output_index = 3 if args.model_type == 'tnlrv3' else 2
        self.bert_config = config_class.from_pretrained(
            args.config_name,
            output_hidden_states=True,
            num_hidden_layers=args.num_teacher_layers if is_teacher else args.num_student_layers)
        self.bert_model = model_class.from_pretrained(
            args.model_name, config=self.bert_config)
        self.bert_trainable_layer = min(
            len(args.teacher_trainable_layer), len(args.student_trainable_layer))

    def forward(self, x):
        '''
            x: batch_size, word_num * 2
            mask: batch_size, word_num
        '''
        batch_size, num_words = x.shape
        num_words = num_words // 2
        text_ids = torch.narrow(x, 1, 0, num_words)
        text_attmask = torch.narrow(x, 1, num_words, num_words)
        result = self.bert_model(text_ids, text_attmask)
        hidden_states = result[self.output_index][-self.bert_trainable_layer:]
        return hidden_states


class UserEncoder(nn.Module):
    def __init__(self, args):
        super(UserEncoder, self).__init__()
        self.args = args
        if args.model == 'NRMS':
            self.multi_head_self_attn = MultiHeadSelfAttention(
                args.news_dim, args.num_attention_heads, 16, 16)
            self.attn = AttentionPooling(
                args.num_attention_heads * 16, args.user_query_vector_dim)
        else:
            self.attn = AttentionPooling(
                args.news_dim, args.user_query_vector_dim)
        self.pad_doc = nn.Parameter(torch.empty(
            1, args.news_dim).uniform_(-1, 1)).type(torch.FloatTensor)

    def forward(self, news_vecs, log_mask=None):
        '''
            news_vecs: batch_size, history_num, news_dim
            log_mask: batch_size, history_num
        '''
        bz = news_vecs.shape[0]
        if self.args.user_log_mask:
            if self.args.model == 'NRMS':
                news_vecs = self.multi_head_self_attn(
                    news_vecs, news_vecs, news_vecs, log_mask)
                user_vec = self.attn(news_vecs, log_mask)
            else:
                user_vec = self.attn(news_vecs, log_mask)
        else:
            padding_doc = self.pad_doc.unsqueeze(dim=0).expand(
                bz, self.args.user_log_length, -1)
            news_vecs = news_vecs * \
                log_mask.unsqueeze(dim=-1) + padding_doc * \
                (1 - log_mask.unsqueeze(dim=-1))
            if self.args.model == 'NRMS':
                news_vecs = self.multi_head_self_attn(
                    news_vecs, news_vecs, news_vecs)
                user_vec = self.attn(news_vecs)
            else:
                user_vec = self.attn(news_vecs)
        return user_vec


def kd_ce_loss(logits_S, logits_T, temperature=1):
    '''
    Calculate the cross entropy between logits_S and logits_T
    :param logits_S: Tensor of shape (batch_size, length, num_labels) or (batch_size, num_labels)
    :param logits_T: Tensor of shape (batch_size, length, num_labels) or (batch_size, num_labels)
    :param temperature: A float or a tensor of shape (batch_size, length) or (batch_size,)
    '''
    beta_logits_T = logits_T / temperature
    beta_logits_S = logits_S
    p_T = F.softmax(beta_logits_T, dim=-1)
    loss = -(p_T * F.log_softmax(beta_logits_S, dim=-1)).sum(dim=-1).mean()
    return loss


def hid_mse_loss(state_S, state_T, mask=None):
    '''
    * Calculates the mse loss between `state_S` and `state_T`, which are the hidden state of the models.
    * If the `inputs_mask` is given, masks the positions where ``input_mask==0``.
    * If the hidden sizes of student and teacher are different, 'proj' option is required in `inetermediate_matches` to match the dimensions.
    :param torch.Tensor state_S: tensor of shape  (*batch_size*, *length*, *hidden_size*)
    :param torch.Tensor state_T: tensor of shape  (*batch_size*, *length*, *hidden_size*)
    :param torch.Tensor mask:    tensor of shape  (*batch_size*, *length*)
    '''
    if mask is None:
        loss = F.mse_loss(state_S, state_T)
    else:
        mask = mask.to(state_S)
        valid_count = mask.sum() * state_S.size(-1)
        loss = (F.mse_loss(state_S, state_T, reduction='none')
                * mask.unsqueeze(-1)).sum() / valid_count
    return loss


def att_mse_loss(attention_S, attention_T, mask=None):
    '''
    * Calculates the mse loss between `attention_S` and `attention_T`.
    * If the `inputs_mask` is given, masks the positions where ``input_mask==0``.
    :param torch.Tensor logits_S: tensor of shape  (*batch_size*, *num_heads*, *length*, *length*)
    :param torch.Tensor logits_T: tensor of shape  (*batch_size*, *num_heads*, *length*, *length*)
    :param torch.Tensor mask: tensor of shape  (*batch_size*, *length*)
    '''
    if mask is None:
        attention_S_select = torch.where(
            attention_S <= -1e-3, torch.zeros_like(attention_S), attention_S)
        attention_T_select = torch.where(
            attention_T <= -1e-3, torch.zeros_like(attention_T), attention_T)
        loss = F.mse_loss(attention_S_select, attention_T_select)
    else:
        mask = mask.to(attention_S).unsqueeze(
            1).expand(-1, attention_S.size(1), -1)  # (bs, num_of_heads, len)
        valid_count = torch.pow(mask.sum(dim=2), 2).sum()
        loss = (F.mse_loss(attention_S, attention_T, reduction='none')
                * mask.unsqueeze(-1) * mask.unsqueeze(2)).sum() / valid_count
    return loss


class Model(torch.nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.args = args
        self.share_pooling = AttentionPooling(
            args.word_embedding_dim, args.news_query_vector_dim)
        self.share_dense = nn.Linear(args.word_embedding_dim, args.news_dim)
        self.teacher = NewsEncoder(args, is_teacher=True)
        self.student = NewsEncoder(args, is_teacher=False)
        self.user_encoder = UserEncoder(args)
        self.target_loss_fn = nn.CrossEntropyLoss()

    def forward(self, batch_news_feature, history_index, candidate_index, label):
        batch_size = history_index.shape[0]
        history_mask = history_index.ne(0).float()
        student_hidden_states = self.student(batch_news_feature)
        student_word_vecs = self.share_pooling(student_hidden_states[-1])
        student_news_vec = self.share_dense(student_word_vecs)
        student_batch_news_vec = torch.cat(
            [self.user_encoder.pad_doc, student_news_vec], dim=0)
        student_history_news_vec = torch.index_select(
            student_batch_news_vec, 0, history_index.reshape(-1)).reshape(batch_size, -1, self.args.news_dim)
        student_candidate_news_vec = torch.index_select(
            student_batch_news_vec, 0, candidate_index.reshape(-1)).reshape(batch_size, -1, self.args.news_dim)
        student_user_vec = self.user_encoder(
            student_history_news_vec, history_mask)
        student_score = torch.bmm(
            student_candidate_news_vec, student_user_vec.unsqueeze(dim=-1)).squeeze(dim=-1)

        teacher_hidden_states = self.teacher(batch_news_feature)
        teacher_word_vecs = self.share_pooling(teacher_hidden_states[-1])
        teacher_news_vec = self.share_dense(teacher_word_vecs)
        teacher_batch_news_vec = torch.cat(
            [self.user_encoder.pad_doc, teacher_news_vec], dim=0)
        teacher_history_news_vec = torch.index_select(
            teacher_batch_news_vec, 0, history_index.reshape(-1)).reshape(batch_size, -1, self.args.news_dim)
        teacher_candidate_news_vec = torch.index_select(
            teacher_batch_news_vec, 0, candidate_index.reshape(-1)).reshape(batch_size, -1, self.args.news_dim)
        teacher_user_vec = self.user_encoder(
            teacher_history_news_vec, history_mask)
        teacher_score = torch.bmm(
            teacher_candidate_news_vec, teacher_user_vec.unsqueeze(dim=-1)).squeeze(dim=-1)

        student_target_loss = self.target_loss_fn(student_score, label)
        teacher_target_loss = self.target_loss_fn(teacher_score, label)
        distill_loss = kd_ce_loss(
            student_score, teacher_score, self.args.temperature)
        emb_loss = hid_mse_loss(student_batch_news_vec, teacher_batch_news_vec) + \
            hid_mse_loss(student_user_vec, teacher_user_vec)

        hidden_loss = 0
        attn_mask = torch.narrow(
            batch_news_feature, -1, self.args.num_words_title, self.args.num_words_title)
        for student_hs, teacher_hs in zip(student_hidden_states, teacher_hidden_states):
            hidden_loss = hidden_loss + \
                hid_mse_loss(student_hs, teacher_hs, attn_mask)

        student_total_loss = distill_loss + self.args.coef * \
            student_target_loss + hidden_loss + emb_loss
        return student_total_loss, teacher_target_loss, student_target_loss, distill_loss, hidden_loss, emb_loss, student_score, teacher_score
