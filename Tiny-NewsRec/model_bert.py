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

        q_s = self.W_Q(Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k_s = self.W_K(K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v_s = self.W_V(V).view(batch_size, -1, self.n_heads, self.d_v).transpose(1, 2)

        context = self.scaled_dot_product_attn(q_s, k_s, v_s, mask)
        output = context.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_v)
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
        self.bert_model = model_class.from_pretrained(args.model_name, config=self.bert_config)
        if args.pooling == 'att':
            self.attn = AttentionPooling(self.bert_config.hidden_size, args.news_query_vector_dim)
        self.dense = nn.Linear(self.bert_config.hidden_size, args.news_dim)

    def forward(self, x):
        '''
            x: batch_size, word_num * 2
            mask: batch_size, word_num
        '''
        batch_size, num_words = x.shape
        num_words = num_words // 2
        text_ids = torch.narrow(x, 1, 0, num_words)
        text_attmask = torch.narrow(x, 1, num_words, num_words)
        word_vecs = self.bert_model(text_ids, text_attmask)[self.output_index][self.bert_config.num_hidden_layers]
        if self.pooling == 'cls':
            news_vec = torch.narrow(word_vecs, 1, 0, 1).squeeze(dim=1)
        elif self.pooling == 'att':
            news_vec = self.attn(word_vecs)
        else:
            news_vec = torch.mean(word_vecs, dim=1)
        news_vec = self.dense(news_vec)
        return news_vec

class UserEncoder(nn.Module):
    def __init__(self, args):
        super(UserEncoder, self).__init__()
        self.args = args
        if args.model == 'NRMS':
            self.multi_head_self_attn = MultiHeadSelfAttention(args.news_dim, args.num_attention_heads, 16, 16)
            self.attn = AttentionPooling(args.num_attention_heads * 16, args.user_query_vector_dim)
        else:
            self.attn = AttentionPooling(args.news_dim, args.user_query_vector_dim)
        self.pad_doc = nn.Parameter(torch.empty(1, args.news_dim).uniform_(-1, 1)).type(torch.FloatTensor)
    
    def forward(self, news_vecs, log_mask=None):
        '''
            news_vecs: batch_size, history_num, news_dim
            log_mask: batch_size, history_num
        '''
        bz = news_vecs.shape[0]
        if self.args.user_log_mask:
            if self.args.model == 'NRMS':
                news_vecs = self.multi_head_self_attn(news_vecs, news_vecs, news_vecs, log_mask)
                user_vec = self.attn(news_vecs, log_mask)
            else:
                user_vec = self.attn(news_vecs, log_mask)
        else:
            padding_doc = self.pad_doc.unsqueeze(dim=0).expand(bz, self.args.user_log_length, -1)
            news_vecs = news_vecs * log_mask.unsqueeze(dim=-1) + padding_doc * (1 - log_mask.unsqueeze(dim=-1))
            if self.args.model == 'NRMS':
                news_vecs = self.multi_head_self_attn(news_vecs, news_vecs, news_vecs)
                user_vec = self.attn(news_vecs)
            else:
                user_vec = self.attn(news_vecs)
        return user_vec

class ModelBert(torch.nn.Module):
    def __init__(self, args, is_teacher):
        super(ModelBert, self).__init__()
        self.args = args
        self.news_encoder = NewsEncoder(args, is_teacher)
        self.user_encoder = UserEncoder(args)

    def forward(self, history, history_mask, candidate):
        '''
            history: batch_size, history_length, num_word_title * 2
            history_mask: batch_size, history_length
            candidate: batch_size, 1+K, num_word_title * 2
        '''
        batch_size = history.shape[0]
        input_id_num = history.shape[-1]
        candidate_news = candidate.reshape(-1, input_id_num)
        candidate_news_vecs = self.news_encoder(candidate_news).reshape(batch_size, -1, self.args.news_dim)

        history_news = history.reshape(-1, input_id_num)
        history_news_vecs = self.news_encoder(history_news).reshape(-1, self.args.user_log_length, self.args.news_dim)

        user_vec = self.user_encoder(history_news_vecs, history_mask)
        score = torch.bmm(candidate_news_vecs, user_vec.unsqueeze(dim=-1)).squeeze(dim=-1)
        return score, history_news_vecs, candidate_news_vecs, user_vec

def kd_ce_loss(logits_S, logits_T, temperature=1):
    '''
    Calculate the cross entropy between logits_S and logits_T
    :param logits_S: Tensor of shape (batch_size, length, num_labels) or (batch_size, num_labels)
    :param logits_T: Tensor of shape (batch_size, length, num_labels) or (batch_size, num_labels)
    :param temperature: A float or a tensor of shape (batch_size, length) or (batch_size,)
    '''
    beta_logits_T = logits_T / temperature
    beta_logits_S = logits_S / temperature
    p_T = F.softmax(beta_logits_T, dim=-1)
    loss = -(p_T * F.log_softmax(beta_logits_S, dim=-1)).sum(dim=-1).mean()
    return loss

def hid_mse_loss(state_S, state_T, mask=None, reduce=True):
    '''
    * Calculates the mse loss between `state_S` and `state_T`, which are the hidden state of the models.
    * If the `inputs_mask` is given, masks the positions where ``input_mask==0``.
    * If the hidden sizes of student and teacher are different, 'proj' option is required in `inetermediate_matches` to match the dimensions.
    :param torch.Tensor state_S: tensor of shape  (*batch_size*, *length*, *hidden_size*)
    :param torch.Tensor state_T: tensor of shape  (*batch_size*, *length*, *hidden_size*)
    :param torch.Tensor mask:    tensor of shape  (*batch_size*, *length*)
    '''
    if mask is None:
        if not reduce:
            loss = F.mse_loss(state_S, state_T, reduction='none').mean(dim=-1)
        else:
            loss = F.mse_loss(state_S, state_T)
    else:
        if not reduce:
            loss = (F.mse_loss(state_S, state_T, reduction='none') * mask.unsqueeze(-1)).mean(dim=-1)
        else:
            valid_count = mask.sum() * state_S.size(-1)
            loss = (F.mse_loss(state_S, state_T, reduction='none') * mask.unsqueeze(-1)).sum() / valid_count
    return loss

class Model(torch.nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.args = args
        self.teachers = nn.ModuleList([UserEncoder(args) for _ in range(args.num_teachers)])
        self.student = ModelBert(args, is_teacher=False)
        self.target_loss_fn = nn.CrossEntropyLoss()
        self.transform_matrix = nn.ModuleList([nn.Linear(args.news_dim, args.news_dim) for _ in range(args.num_teachers)])
        self.W = nn.Parameter(torch.empty(1)).type(torch.FloatTensor)

        nn.init.constant_(self.W, 1)
        for module in self.transform_matrix:
            nn.init.xavier_uniform_(module.weight, gain=1.)
            nn.init.constant_(module.bias, 0.0)

    def forward(self, history, history_mask, candidate, label, teacher_history_embs, teacher_candidate_embs):
        '''
            teacher_history_embs: [(batch_size, user_log_length, news_emb) * num_teachers]
            teacher_candidate_emb: [(batch_size, 1+K, news_emb) * num_teachers]
        '''
        student_score, student_history_emb, student_candidate_emb, student_user_emb = self.student(history, history_mask, candidate)
        student_news_emb = torch.cat([student_history_emb, student_candidate_emb], dim=1)
        target_loss = self.target_loss_fn(student_score, label)

        teacher_scores, teacher_losses = [], []
        NE_MSEs, UE_MSEs = [], []
        for i, (teacher_history, teacher_candidate) in enumerate(zip(teacher_history_embs, teacher_candidate_embs)):
            teacher_news_emb = torch.cat([teacher_history, teacher_candidate], dim=1)
            teacher_news_emb_proj = self.transform_matrix[i](teacher_news_emb)
            NE_MSEs.append(hid_mse_loss(student_news_emb, teacher_news_emb_proj, reduce=False).mean(dim=-1))

            teacher_user_vector = self.teachers[i](teacher_history, history_mask)
            teacher_user_vector_proj = self.transform_matrix[i](teacher_user_vector)
            UE_MSEs.append(hid_mse_loss(student_user_emb, teacher_user_vector_proj, reduce=False))
            
            score = torch.bmm(teacher_candidate, teacher_user_vector.unsqueeze(dim=-1)).squeeze(dim=-1)
            teacher_loss = F.cross_entropy(score, label, reduction='none')
            teacher_scores.append(score)
            teacher_losses.append(teacher_loss)

        teacher_losses = -torch.stack(teacher_losses, dim=-1) * self.W
        teacher_weights = F.softmax(teacher_losses, dim=-1)

        teacher_scores = torch.stack(teacher_scores, dim=-1)
        teacher_scores = torch.bmm(teacher_scores, teacher_weights.unsqueeze(dim=-1)).squeeze(dim=-1)
        distill_loss = kd_ce_loss(student_score, teacher_scores, self.args.temperature)

        NE_MSEs = torch.stack(NE_MSEs, dim=-1)
        UE_MSEs = torch.stack(UE_MSEs, dim=-1)
        emb_loss = (NE_MSEs * teacher_weights).sum(dim=-1).mean() + (UE_MSEs * teacher_weights).sum(dim=-1).mean()
    
        total_loss = distill_loss + self.args.coef * target_loss + emb_loss
        return total_loss, distill_loss, emb_loss, target_loss, student_score