import torch.nn as nn
import torch
from torch.nn import functional as F
from torch.autograd import Variable
from transformers import (BertPreTrainedModel, BertModel,
                          RobertaModel, RobertaConfig,ROBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP,
                          DebertaPreTrainedModel, DebertaConfig, DebertaModel, DEBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP,
                          XLNetPreTrainedModel, XLNetConfig, XLNetModel, XLNET_PRETRAINED_CONFIG_ARCHIVE_MAP,  DEBERTA_PRETRAINED_MODEL_ARCHIVE_LIST)


class BertForListRank(BertPreTrainedModel):
    r"""
    Inputs:
        **input_ids**: ``torch.LongTensor`` of shape ``(batch_size, list_len, seq_len)``:
            Indices of input sequence tokens in the vocabulary.
            The second dimension of the input (`list_len`) indicates the number of choices to score.
        **token_type_ids**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size, list_len, seq_len)``:
            Segment token indices to indicate first and second portions of the inputs.
            The second dimension of the input (`list_len`) indicates the number of choices to score.
            Indices are selected in ``[0, 1]``: ``0`` corresponds to a `sentence A` token, ``1``
        **attention_mask**: (`optional`) ``torch.FloatTensor`` of shape ``(batch_size, list_len, seq_len)``:
            Mask to avoid performing attention on padding token indices.
            The second dimension of the input (`list_len`) indicates the number of choices to score.
            Mask values selected in ``[0, 1]``:
            ``1`` for tokens that are NOT MASKED, ``0`` for MASKED tokens.
        **head_mask**: (`optional`) ``torch.FloatTensor`` of shape ``(num_heads,)`` or ``(num_layers, num_heads)``:
            Mask to nullify selected heads of the self-attention modules.
            Mask values selected in ``[0, 1]``:
            ``1`` indicates the head is **not masked**, ``0`` indicates the head is **masked**.
        **labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
            Labels for computing the multiple choice classification loss.
            Indices should be in ``[0, ..., list_len]`` where `list_len` is the size of the second dimension
            of the input tensors. (see `input_ids` above)

    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **logits**: ``torch.FloatTensor`` of shape ``(batch_size, list_len)`` where `list_len` is the size of
            the second dimension of the input tensors. (see `input_ids` above).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, seq_len, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, seq_len, seq_len)``:
            Attentions weights after softmax, used to compute the weighted average in the self-attention heads.
    """

    def __init__(self, config):
        super(BertForListRank, self).__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.linear_dropout_prob)
        self.linear = nn.Linear(config.hidden_size, 1)

        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, labels=None):
        batch_size, list_len, seq_len = input_ids.shape

        # (batch_size * list_len, seq_len)
        flat_input_ids = input_ids.view(-1, seq_len)
        flat_position_ids = position_ids.view(-1, seq_len) if position_ids is not None else None
        flat_token_type_ids = token_type_ids.view(-1, seq_len) if token_type_ids is not None else None
        flat_attention_mask = attention_mask.view(-1, seq_len) if attention_mask is not None else None

        if flat_attention_mask is not None:
            # (batch_size * list_len,)
            seq_lens = flat_attention_mask.sum(dim=-1)
            _, sorted_seq_indices = seq_lens.sort(descending=True)
            # 根据flat_attention_mask的大小对序列进行排序，也就是字符串的长度
            # (batch_size * list_len, seq_len)
            flat_input_ids = flat_input_ids[sorted_seq_indices]
            flat_attention_mask = flat_attention_mask[sorted_seq_indices]
            if flat_position_ids is not None:
                flat_position_ids = flat_position_ids[sorted_seq_indices]
            if flat_token_type_ids is not None:
                flat_token_type_ids = flat_token_type_ids[sorted_seq_indices]

            # 序列总共的长度，以及最大的长度
            real_seq_num = (seq_lens > 0).sum().item()
            max_seq_len = seq_lens.max().long().item()
            # (real_seq_num, max_seq_len)
            flat_input_ids = flat_input_ids[:real_seq_num, :max_seq_len]
            flat_attention_mask = flat_attention_mask[:real_seq_num, :max_seq_len]
            if flat_position_ids is not None:
                flat_position_ids = flat_position_ids[:real_seq_num, :max_seq_len]
            if flat_token_type_ids is not None:
                flat_token_type_ids = flat_token_type_ids[:real_seq_num, :max_seq_len]

        outputs = self.bert(flat_input_ids,
                            position_ids=flat_position_ids,
                            token_type_ids=flat_token_type_ids,
                            attention_mask=flat_attention_mask, head_mask=head_mask)
        hidden_states = outputs[0]  # (real_seq_num, max_seq_len, hidden_size)

        # pooled_output = outputs[1]
        pooled_output = hidden_states.mean(dim=1)  # (real_seq_num, hidden_size)
        pooled_output = self.dropout(pooled_output)
        logits = self.linear(pooled_output).squeeze(-1)  # (real_seq_num,)

        if flat_attention_mask is not None:
            # (batch_size * list_len,)
            logits = F.pad(logits, mode='constant', value=float('-inf'), pad=[0, batch_size * list_len - real_seq_num])
            _, unsorted_seq_indices = sorted_seq_indices.sort()
            logits = logits[unsorted_seq_indices]

        # (batch_size, list_len)
        logits = logits.view(batch_size, list_len)

        return logits


class RobertaForListRank(BertPreTrainedModel):
    config_class = RobertaConfig
    # pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "roberta"

    def __init__(self, config):
        super(RobertaForListRank, self).__init__(config)
        
        self.config = config
        self.roberta = RobertaModel(config)
        self.pla = PLAttention(config, 8, 3)
        self.linear = nn.Linear(config.hidden_size, 1)
        self.dropout = nn.Dropout(config.linear_dropout_prob)
        self.init_weights()
    
    def forward(self, device, input_ids, token_type_ids=None, attention_mask=None, position_ids=None, head_mask=None):
        batch_size, list_len, seq_len = input_ids.shape
        
        # (batch_size * list_len, seq_len)
        flat_input_ids = input_ids.view(-1, seq_len)
        flat_position_ids = position_ids.view(-1, seq_len) if position_ids is not None else None
        flat_token_type_ids = token_type_ids.view(-1, seq_len) if token_type_ids is not None else None
        flat_attention_mask = attention_mask.view(-1, seq_len) if attention_mask is not None else None

        if flat_attention_mask is not None:
            # (batch_size * list_len,)
            seq_lens = flat_attention_mask.sum(dim=-1)
            _, sorted_seq_indices = seq_lens.sort(descending=True)

            # (batch_size * list_len, seq_len)
            flat_input_ids = flat_input_ids[sorted_seq_indices]
            flat_attention_mask = flat_attention_mask[sorted_seq_indices]
            if flat_position_ids is not None:
                flat_position_ids = flat_position_ids[sorted_seq_indices]
            if flat_token_type_ids is not None:
                flat_token_type_ids = flat_token_type_ids[sorted_seq_indices]

            real_seq_num = (seq_lens > 0).sum().item()
            max_seq_len = seq_lens.max().long().item()
            # (real_seq_num, max_seq_len)
            flat_input_ids = flat_input_ids[:real_seq_num, :max_seq_len]
            flat_attention_mask = flat_attention_mask[:real_seq_num, :max_seq_len]
            if flat_position_ids is not None:
                flat_position_ids = flat_position_ids[:real_seq_num, :max_seq_len]
            if flat_token_type_ids is not None:
                flat_token_type_ids = flat_token_type_ids[:real_seq_num, :max_seq_len]


        outputs = self.roberta(flat_input_ids,
                               position_ids=flat_position_ids,
                               # token_type_ids=flat_token_type_ids,
                               attention_mask=flat_attention_mask, head_mask=head_mask)
        hidden_states = outputs[0]  # (real_seq_num, max_seq_len, hidden_size)  n(n<=22)* 72 * 1024
        hidden_states = self.pla(hidden_states, device)

        pooled_output = hidden_states.mean(dim=1)  # (real_seq_num, hidden_size) n(n<=22) * 1024

        logits = self.linear(pooled_output).squeeze()  # (real_seq_num,)
        logits = self.dropout(logits)
        if flat_attention_mask is not None:

            logits = F.pad(logits, mode='constant', value=float('-inf'), pad=[0, batch_size * list_len - real_seq_num])

            _, unsorted_seq_indices = sorted_seq_indices.sort()
            logits = logits[unsorted_seq_indices]

        logits = logits.view(batch_size, list_len)

        return logits

class PLAttention(nn.Module):
    def __init__(self, config, heads, window_size):
        super(PLAttention, self).__init__()
        self.window_size = window_size
        self.attention = Attention(
            config,
            heads,
            window_size,
        )

    def forward(self, M, device):
        window_size = self.window_size
        group_num, index_one, index_two, seq_len = self.prepare(M, window_size, device)

        for batch in range(window_size):
            index = (index_one - batch) % seq_len
            M = M[:, index]

            M = self.segmentation(M, group_num, window_size, step=1)
            M = self.attention(M, M, M)
            M = self.segmentation(M, group_num, window_size, step=2)

            if batch != 0:
                index_two = torch.cat((index_two[1:], index_two[:1]), 0)
            M = M[:, index_two]

        M = M[:, :seq_len]
        return M

    def segmentation(self, M, group_num, window_size, step):
        if step == 1:
            N = M[:, 0:group_num]
            for i in range(window_size-1):
                N = torch.cat((N, M[:, group_num*(i+1):group_num*(i+2)]), 2)
        if step == 2:
            hidden_size = M.shape[2] // window_size
            N = M[:, :, 0:hidden_size]
            for i in range(window_size-1):
                N = torch.cat((N, M[:, :, hidden_size*(i+1):hidden_size*(i+2)]), 1)
        return N

    def prepare(self, test, window_size, device):
        # 前期准备
        seq_len = divisible_len = test.shape[1]
        if divisible_len % window_size:
            divisible_len = window_size - (divisible_len % window_size) + divisible_len  # 可被window_size整除
        group_num = divisible_len // window_size  # 组数

        # 获取重组的index
        index_init = torch.arange(0, divisible_len)
        mask_all = index_init % window_size
        mask_list = []
        for i in range(window_size):
            mask = mask_all == i
            mask_index = index_init[mask]
            mask_list.append(mask_index)
        index_one = mask_list[0]
        for i in range(window_size - 1):
            index_one = torch.cat((index_one, mask_list[i + 1]), 0)

        # 获取还原的index
        base_group_list = []
        for i in range(window_size):
            base_group_list.append(group_num * i)
        base_group_list = torch.tensor(base_group_list)
        group_list = []
        for i in range(group_num):
            group_list.append(base_group_list + i)
        index_two = group_list[0]
        for i in range(group_num - 1):
            index_two = torch.cat((index_two, group_list[i + 1]), 0)

        return group_num, index_one, index_two, seq_len


class Attention(nn.Module):
    def __init__(self, config, heads, window_size=None):
        super(Attention, self).__init__()
        self.embed_size = config.hidden_size*window_size if window_size else config.hidden_size
        self.heads = heads
        self.head_dim = self.embed_size // heads

        assert (self.head_dim * heads == self.embed_size), "Embed size needs to be div by heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, self.embed_size)

    def forward(self, values, keys, query):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # split embedding into self.heads pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        energy = torch.einsum("nqhd,nkhd->nhqk", queries, keys)
        # queries shape: (N, query_len, heads, heads_dim)
        # keys shape : (N, key_len, heads, heads_dim)
        # energy shape: (N, heads, query_len, key_len)

        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)
        out = torch.einsum("nhql, nlhd->nqhd", [attention, values]).reshape(N, query_len, self.heads * self.head_dim)
        # attention shape: (N, heads, query_len, key_len)
        # values shape: (N, value_len, heads, heads_dim)
        # (N, query_len, heads, head_dim)

        out = self.fc_out(out)
        return out


class DebertaForListRank(DebertaPreTrainedModel):
    config_class = DebertaConfig
    base_model_prefix = "deberta"

    def __init__(self, config):
        super(DebertaForListRank, self).__init__(config)

        self.deberta = DebertaModel(config)
        self.dropout = nn.Dropout(config.linear_dropout_prob)
        self.linear = nn.Linear(config.hidden_size, 1)
        self.init_weights()
    #     self.lstm = nn.LSTM(input_size=config.hidden_size, hidden_size=config.hidden_size//2, bidirectional=True)
    #     self.hidden = self.rand_init_hidden(1)
    #
    # def rand_init_hidden(self, batch_size):
    #     """
    #     random initialize hidden variable
    #     """
    #     return Variable(
    #         torch.randn(2, batch_size, self.config.hidden_size//2)).cuda(), Variable(
    #         torch.randn(2, batch_size, self.config.hidden_size//2)).cuda()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, position_ids=None):
        batch_size, list_len, seq_len = input_ids.shape

        # (batch_size * list_len, seq_len)
        flat_input_ids = input_ids.view(-1, seq_len)
        flat_position_ids = position_ids.view(-1, seq_len) if position_ids is not None else None
        flat_token_type_ids = token_type_ids.view(-1, seq_len) if token_type_ids is not None else None
        flat_attention_mask = attention_mask.view(-1, seq_len) if attention_mask is not None else None

        if flat_attention_mask is not None:
            # (batch_size * list_len,)
            seq_lens = flat_attention_mask.sum(dim=-1)
            _, sorted_seq_indices = seq_lens.sort(descending=True)

            # (batch_size * list_len, seq_len)
            flat_input_ids = flat_input_ids[sorted_seq_indices]
            flat_attention_mask = flat_attention_mask[sorted_seq_indices]
            if flat_position_ids is not None:
                flat_position_ids = flat_position_ids[sorted_seq_indices]
            if flat_token_type_ids is not None:
                flat_token_type_ids = flat_token_type_ids[sorted_seq_indices]

            real_seq_num = (seq_lens > 0).sum().item()
            max_seq_len = seq_lens.max().long().item()
            # (real_seq_num, max_seq_len)
            flat_input_ids = flat_input_ids[:real_seq_num, :max_seq_len]
            flat_attention_mask = flat_attention_mask[:real_seq_num, :max_seq_len]
            if flat_position_ids is not None:
                flat_position_ids = flat_position_ids[:real_seq_num, :max_seq_len]
            if flat_token_type_ids is not None:
                flat_token_type_ids = flat_token_type_ids[:real_seq_num, :max_seq_len]
        outputs = self.deberta(flat_input_ids,
                               position_ids=flat_position_ids,
                               # token_type_ids=flat_token_type_ids,
                               attention_mask=flat_attention_mask)
        hidden_states = outputs[0]  # (real_seq_num, max_seq_len, hidden_size)

        pooled_output = hidden_states.mean(dim=1)  # (real_seq_num, hidden_size)
        # pooled_output = self.dropout(pooled_output)

        # outputs, _ = self.lstm(pooled_output.unsqueeze(1), self.hidden)

        logits = self.linear(pooled_output).squeeze(-1)  # (real_seq_num,)
        logits = self.dropout(logits)
        if flat_attention_mask is not None:
            # (batch_size * list_len,)
            logits = F.pad(logits, mode='constant', value=float('-inf'), pad=[0, batch_size * list_len - real_seq_num])
            _, unsorted_seq_indices = sorted_seq_indices.sort()
            logits = logits[unsorted_seq_indices]

        # (batch_size, list_len)
        logits = logits.view(batch_size, list_len)

        return logits

class XLNetForListRank(XLNetPreTrainedModel):
    r"""
    Inputs:
        **input_ids**: ``torch.LongTensor`` of shape ``(batch_size, list_len, seq_len)``:
            Indices of input sequence tokens in the vocabulary.
            The second dimension of the input (`list_len`) indicates the number of choices to score.
        **token_type_ids**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size, list_len, seq_len)``:
            Segment token indices to indicate first and second portions of the inputs.
            The second dimension of the input (`list_len`) indicates the number of choices to score.
            Indices are selected in ``[0, 1]``: ``0`` corresponds to a `sentence A` token, ``1``
        **attention_mask**: (`optional`) ``torch.FloatTensor`` of shape ``(batch_size, list_len, seq_len)``:
            Mask to avoid performing attention on padding token indices.
            The second dimension of the input (`list_len`) indicates the number of choices to score.
            Mask values selected in ``[0, 1]``:
            ``1`` for tokens that are NOT MASKED, ``0`` for MASKED tokens.
        **head_mask**: (`optional`) ``torch.FloatTensor`` of shape ``(num_heads,)`` or ``(num_layers, num_heads)``:
            Mask to nullify selected heads of the self-attention modules.
            Mask values selected in ``[0, 1]``:
            ``1`` indicates the head is **not masked**, ``0`` indicates the head is **masked**.
        **labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
            Labels for computing the multiple choice classification loss.
            Indices should be in ``[0, ..., list_len]`` where `list_len` is the size of the second dimension
            of the input tensors. (see `input_ids` above)

    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **logits**: ``torch.FloatTensor`` of shape ``(batch_size, list_len)`` where `list_len` is the size of
            the second dimension of the input tensors. (see `input_ids` above).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, seq_len, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, seq_len, seq_len)``:
            Attentions weights after softmax, used to compute the weighted average in the self-attention heads.
    """
    config_class = XLNetConfig
    base_model_prefix = "xlnet"

    def __init__(self, config):
        super(XLNetForListRank, self).__init__(config)

        self.xlnet = XLNetModel(config)
        self.dropout = nn.Dropout(config.linear_dropout_prob)
        self.linear = nn.Linear(config.hidden_size, 1)

        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        position_ids = None
        batch_size, list_len, seq_len = input_ids.shape
        # (batch_size * list_len, seq_len)
        flat_input_ids = input_ids.view(-1, seq_len)
        flat_position_ids = position_ids.view(-1, seq_len) if position_ids is not None else None
        flat_token_type_ids = token_type_ids.view(-1, seq_len) if token_type_ids is not None else None
        flat_attention_mask = attention_mask.view(-1, seq_len) if attention_mask is not None else None

        if flat_attention_mask is not None:
            # (batch_size * list_len,)
            seq_lens = flat_attention_mask.sum(dim=-1)
            _, sorted_seq_indices = seq_lens.sort(descending=True)

            # (batch_size * list_len, seq_len)
            flat_input_ids = flat_input_ids[sorted_seq_indices]
            flat_attention_mask = flat_attention_mask[sorted_seq_indices]
            if flat_token_type_ids is not None:
                flat_token_type_ids = flat_token_type_ids[sorted_seq_indices]

            real_seq_num = (seq_lens > 0).sum().item()
            max_seq_len = seq_lens.max().long().item()
            # (real_seq_num, max_seq_len)
            flat_input_ids = flat_input_ids[:real_seq_num, :max_seq_len]
            flat_attention_mask = flat_attention_mask[:real_seq_num, :max_seq_len]
            if flat_token_type_ids is not None:
                flat_token_type_ids = flat_token_type_ids[:real_seq_num, :max_seq_len]

        '''
        chunk_size = 24
        chunk_input_ids = flat_input_ids.split(chunk_size, dim=0)
        if flat_position_ids is not None:
            chunk_position_ids = flat_position_ids.split(chunk_size, dim=0)
        else:
            chunk_position_ids = [None] * len(chunk_input_ids)
        if flat_token_type_ids is not None:
            chunk_token_type_ids = flat_token_type_ids.split(chunk_size, dim=0)
        else:
            chunk_token_type_ids = [None] * len(chunk_input_ids)
        if flat_attention_mask is not None:
            chunk_attention_mask = flat_attention_mask.split(chunk_size, dim=0)
        else:
            chunk_attention_mask = [None] * len(chunk_input_ids)
        chunk_hidden_states = []
        for i in range(len(chunk_input_ids)):
            outputs = self.roberta(chunk_input_ids[i],
                                   position_ids=chunk_position_ids[i],
                                   # token_type_ids=chunk_token_type_ids[i],
                                   attention_mask=chunk_attention_mask[i], head_mask=head_mask)
            chunk_hidden_states.append(outputs[0])  # (chunk_size, max_seq_len, hidden_size)
        hidden_states = torch.cat(chunk_hidden_states, dim=0)  # (real_seq_num, max_seq_len, hidden_size)
        '''

        outputs = self.xlnet(flat_input_ids,
                               token_type_ids=flat_token_type_ids,
                               attention_mask=flat_attention_mask)
        hidden_states = outputs[0]  # (real_seq_num, max_seq_len, hidden_size)

        # pooled_output = outputs[1]
        pooled_output = hidden_states.mean(dim=1)  # (real_seq_num, hidden_size)
        pooled_output = self.dropout(pooled_output)
        logits = self.linear(pooled_output).squeeze(-1)  # (real_seq_num,)

        if flat_attention_mask is not None:
            # (batch_size * list_len,)
            logits = F.pad(logits, mode='constant', value=float('-inf'), pad=[0, batch_size * list_len - real_seq_num])
            _, unsorted_seq_indices = sorted_seq_indices.sort()
            logits = logits[unsorted_seq_indices]

        # (batch_size, list_len)
        logits = logits.view(batch_size, list_len)

        return logits




