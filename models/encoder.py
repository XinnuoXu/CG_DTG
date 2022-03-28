import math
import torch
import torch.nn as nn

from models.neural import MultiHeadedAttention, PositionwiseFeedForward


class Classifier(nn.Module):
    def __init__(self, hidden_size):
        super(Classifier, self).__init__()
        self.linear1 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, mask_cls):
        h = self.linear1(x).squeeze(-1)
        sent_scores = self.sigmoid(h) * mask_cls.float()
        return sent_scores


class PositionalEncoding(nn.Module):

    def __init__(self, dropout, dim, max_len=5000):
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, dim, 2, dtype=torch.float) *
                              -(math.log(10000.0) / dim)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.unsqueeze(0)
        super(PositionalEncoding, self).__init__()
        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(p=dropout)
        self.dim = dim

    def forward(self, emb, step=None):
        emb = emb * math.sqrt(self.dim)
        if (step):
            emb = emb + self.pe[:, step][:, None, :]

        else:
            emb = emb + self.pe[:, :emb.size(1)]
        emb = self.dropout(emb)
        return emb

    def get_emb(self, emb):
        return self.pe[:, :emb.size(1)]


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, heads, d_ff, dropout):
        super(TransformerEncoderLayer, self).__init__()

        self.self_attn = MultiHeadedAttention(
            heads, d_model, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, iter, query, inputs, mask):
        if (iter != 0):
            input_norm = self.layer_norm(inputs)
        else:
            input_norm = inputs

        mask = mask.unsqueeze(1)
        context, _ = self.self_attn(input_norm, input_norm, input_norm,
                                 mask=mask)
        out = self.dropout(context) + inputs
        return self.feed_forward(out)


class TMTLayer(nn.Module):
    def __init__(self, d_model,  d_ff, dropout, iter):
        super(TMTLayer, self).__init__()

        self.iter = iter
        self.self_attn = StructuredAttention( d_model, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.dropout = nn.Dropout(dropout)
        self.linears1 = nn.ModuleList([nn.Linear(2*d_model,d_model) for _ in range(iter)])
        self.relu = nn.ReLU()
        self.linears2 = nn.ModuleList([nn.Linear(d_model,d_model) for _ in range(iter)])
        self.layer_norm = nn.ModuleList([nn.LayerNorm(d_model, eps=1e-6) for _ in range(iter)])

    def forward(self, x, structure_vec, mask):
        vecs = [x]

        mask = mask.unsqueeze(1)
        attn, root = self.self_attn(structure_vec,mask=mask)
        for i in range(self.iter):
            context = torch.matmul(attn, vecs[-1])
            new_c = self.linears2[i](self.relu(self.linears1[i](torch.cat([vecs[-1], context], -1))))
            new_c = self.layer_norm[i](new_c)
            vecs.append(new_c)

        return vecs[-1], root, attn


class StructuredAttention(nn.Module):
    def __init__(self, model_dim, dropout=0.1):
        self.model_dim = model_dim

        super(StructuredAttention, self).__init__()

        self.linear_keys = nn.Linear(model_dim, self.model_dim)
        self.linear_query = nn.Linear(model_dim, self.model_dim)
        self.linear_root = nn.Linear(model_dim, 1)
        self.dropout = nn.Dropout(dropout)

    def _getMatrixTree_multi(self, scores, root):
        A = scores.exp()
        R = root.exp()

        L = torch.sum(A, 1)
        L = torch.diag_embed(L)
        L = L - A
        LL = L + torch.diag_embed(R)
        LL_inv = torch.inverse(LL)  # batch_l, doc_l, doc_l
        LL_inv_diag = torch.diagonal(LL_inv, 0, 1, 2)
        d0 = R * LL_inv_diag
        LL_inv_diag = torch.unsqueeze(LL_inv_diag, 2)

        _A = torch.transpose(A, 1, 2)
        _A = _A * LL_inv_diag
        tmp1 = torch.transpose(_A, 1, 2)
        tmp2 = A * torch.transpose(LL_inv, 1, 2)

        d = tmp1 - tmp2
        return d, d0


    def forward(self, x, mask=None):

        key = self.linear_keys(x)
        query = self.linear_query(x)
        root = self.linear_root(x).squeeze(-1)

        query = query / math.sqrt(self.model_dim)
        scores = torch.matmul(query, key.transpose(1, 2))

        mask = mask.float()
        root = root - mask.squeeze(1) * 50
        root = torch.clamp(root, min=-40)
        scores = scores - mask * 50
        scores = scores - torch.transpose(mask, 1, 2) * 50
        scores = torch.clamp(scores, min=-40)
        # _logits = _logits + (tf.transpose(bias, [0, 2, 1]) - 1) * 40
        # _logits = tf.clip_by_value(_logits, -40, 1e10)

        d, d0 = self._getMatrixTree_multi(scores, root)
        attn = torch.transpose(d, 1,2)
        if mask is not None:
            mask = mask.expand_as(scores).bool()
            attn = attn.masked_fill(mask, 0)
        return attn, d0


class SentenceClassification(nn.Module):
    def __init__(self, d_model, d_ff, heads, dropout, num_inter_layers=0):
        super(ExtTransformerEncoder, self).__init__()
        self.d_model = d_model
        self.num_inter_layers = num_inter_layers
        self.pos_emb = PositionalEncoding(dropout, d_model)
        self.transformer_inter = nn.ModuleList(
            [TransformerEncoderLayer(d_model, heads, d_ff, dropout)
             for _ in range(num_inter_layers)])
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.wo = nn.Linear(d_model, 1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, top_vecs, mask):
        """ See :obj:`EncoderBase.forward()`"""

        batch_size, n_sents = top_vecs.size(0), top_vecs.size(1)
        pos_emb = self.pos_emb.pe[:, :n_sents]
        x = top_vecs * mask[:, :, None].float()
        x = x + pos_emb

        for i in range(self.num_inter_layers):
            x = self.transformer_inter[i](i, x, x, ~ mask)  # all_sents * max_tokens * dim

        x = self.layer_norm(x)
        sent_scores = self.sigmoid(self.wo(x))
        sent_scores = sent_scores.squeeze(-1) * mask.float()
        sent_scores = sent_scores.squeeze(-1)

        return sent_scores


class TreeInference(nn.Module):
    def __init__(self, d_model, d_ff, dropout, num_inter_layers=0):
        super(TreeInference, self).__init__()
        self.d_model = d_model
        self.num_inter_layers = num_inter_layers
        self.pos_emb = PositionalEncoding(dropout, int(d_model))
        self.transformer_inter = nn.ModuleList([TMTLayer(d_model, d_ff, dropout, i) for i in range(num_inter_layers)])
        self.layer_norm2 = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, sent_vec, mask_block):

        batch_size, n_blocks, _ = sent_vec.size()

        global_pos_emb = self.pos_emb.pe[:, :n_blocks]
        sent_vec = sent_vec + global_pos_emb

        sent_vec = self.layer_norm2(sent_vec)* mask_block.unsqueeze(-1).float()
        structure_vec = sent_vec

        roots = []; structure_vecs = []; attns = []
        for i in range(self.num_inter_layers):
            structure_vec, root, attn = self.transformer_inter[i](sent_vec, structure_vec, ~ mask_block)
            roots.append(root)
            attns.append(attn)
            structure_vecs.append(structure_vec)

        return roots, attns
