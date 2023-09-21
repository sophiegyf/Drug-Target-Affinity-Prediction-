import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.functional as F
from arguments import argparser, logging
from datahelper import *

class ScaledDotProductAttention(nn.Module):
  """Scaled dot-product attention mechanism."""

  def __init__(self, attention_dropout=0.0):
    """Init.
    Args:
      attention_dropout: A scalar, dropout rate.
    """
    super(ScaledDotProductAttention, self).__init__()
    self.dropout = nn.Dropout(attention_dropout)
    self.softmax = nn.Softmax(dim=2)

  def forward(self, q, k, v, scale=None, attn_mask=None):
    """Forward pass.
    Args:
      q: Queries tensor, with shape of [B, L_q, D_q]
      k: Keys tensor, with shape of [B, L_k, D_k]
      v: Values tensor, with shape of [B, L_v, D_v]
      scale: A scalar, scale factor.
      attn_mask: A binary masking tensor, with shape of [B, L_q, L_k]
    Returns:
      Context and attention tensor.
    """
    attention = torch.bmm(q, k.transpose(1, 2))
    if scale:
      attention = attention * scale
    if attn_mask:
      # Mask out attention
      # set a negative infnite to where were padded a `PAD`
      attention = attention.masked_fill_(attn_mask, -np.inf)
    attention = self.softmax(attention)
    attention = self.dropout(attention)
    context = torch.bmm(attention, v)
    return context, attention


class LayerNorm(nn.Module):
  """Layer Normalization. PyTorch has already implemented this `nn.LayerNorm`"""

  def __init__(self, model_dim, epsilon=1e-6):
    """Init.
    Args:
      model_dim: Number of features, or model's dimension.
      epsilon: A small number to avoid numeric error.
    """
    super(LayerNorm, self).__init__()
    # weights
    self.gamma = nn.Parameter(torch.ones(model_dim))
    # bias
    self.beta = nn.Parameter(torch.zeros(model_dim))
    self.epsilon = epsilon

  def forward(self, x):
    """Forward pass.
    Args:
      x: Input tensor, with shape of [B, L, D]
    Returns:
      Normalized input tensor.
    """
    mean = x.mean(-1, keepdim=True)
    std = x.std(-1, keepdim=True)
    return self.gamma * (x - mean) / (std + self.epsilon) + self.beta


class PositionalEncoding(nn.Module):
  """Positional encoding.
  This class is modified from https://github.com/JayParks/transformer/blob/master/transformer/modules.py.
  """

  def __init__(self, model_dim, max_seq_len):
    """Init.
    Args:
      model_dim: The model's dimension,
        the last dimension of input x with shape of [B, L, D]
      max_seq_len: The maximum sequence length, or the time-steps,
        the second dimension of input x with shape of [B, L, D]
    """
    super(PositionalEncoding, self).__init__()

    # j//2 because we have sin and cos tow channels
    position_encoding = np.array([
      [pos / np.pow(10000, 2.0 * (j // 2) / model_dim) for j in range(model_dim)]
      for pos in range(max_seq_len)])
    position_encoding[:, 0::2] = np.sin(position_encoding[:, 0::2])
    position_encoding[:, 1::2] = np.cos(position_encoding[:, 1::2])

    pad_row = torch.zeros([1, model_dim])
    position_encoding = torch.cat((pad_row, position_encoding))
    # additional PAD position index
    self.position_encoding = nn.Embedding(max_seq_len + 1, model_dim)

    self.position_encoding.weight = nn.Parameter(position_encoding,
                                                 requires_grad=False)

  def forward(self, input_len):
    """Forward pass.
    Args:
      input_len: A tensor with shape [B, 1]. Each element's value in ths tensor
        is the length of a sequence from a mini batch.
    Returns:
      Position encoding(or position embedding) of a mini batch sequence.
    """
    max_len = torch.max(input_len)
    tensor = torch.cuda.LongTensor if input_len.is_cuda else torch.LongTensor
    # we start from 1 because 0 is for PAD
    # we pad a sequence with PAD(0) to the max length
    input_pos = tensor(
      [list(range(1, len + 1)) + [0] * (max_len - len) for len in input_len])
    return self.position_encoding(input_pos)


class MultiHeadAttention(nn.Module):
  """Multi-Head attention."""

  def __init__(self, model_dim=512, num_heads=8, dropout=0.0):
    """Init.
    Args:
      model_dim: Model's dimension, default is 512 according to the paper
      num_heads: Number of heads, default is 8 according to the paper
      dropout: Dropout rate for dropout layer
    """
    super(MultiHeadAttention, self).__init__()

    self.dim_per_head = model_dim // num_heads
    self.num_heads = num_heads
    self.linear_k = nn.Linear(model_dim, self.dim_per_head * num_heads)
    self.linear_v = nn.Linear(model_dim, self.dim_per_head * num_heads)
    self.linear_q = nn.Linear(model_dim, self.dim_per_head * num_heads)

    self.dot_product_attention = ScaledDotProductAttention(dropout)
    self.linear_final = nn.Linear(model_dim, model_dim)
    self.dropout = nn.Dropout(dropout)
    self.layer_norm = nn.LayerNorm(model_dim)

  def forward(self, key, value, query, attn_mask=None):
    """Forward pass.
    Args:
      key: Key tensor, with shape of [B, L_k, D]
      value: Value tensor, with shape of [B, L_v, D]
      query: Query tensor, with shape of [B, L_q, D]
      attn_mask: Mask tensor for attention, with shape of [B, L, L]
    """
    residual = query

    dim_per_head = self.dim_per_head
    num_heads = self.num_heads
    batch_size = key.size(0)

    # linear projection
    key = self.linear_k(key)
    value = self.linear_v(value)
    query = self.linear_q(query)

    # split by heads
    key = key.view(batch_size * num_heads, -1, dim_per_head)
    value = value.view(batch_size * num_heads, -1, dim_per_head)
    query = query.view(batch_size * num_heads, -1, dim_per_head)

    if attn_mask:
      attn_mask = attn_mask.repeat(num_heads, 1, 1)
    # scaled dot product attention
    scale = (key.size(-1) // num_heads) ** -0.5
    context, attention = self.dot_product_attention(
      query, key, value, scale, attn_mask)

    # concat heads
    context = context.view(batch_size, -1, dim_per_head * num_heads)

    # final linear projection
    output = self.linear_final(context)

    # dropout
    output = self.dropout(output)

    # add residual and norm layer
    output = self.layer_norm(residual + output)

    return output, attention


class PositionalWiseFeedForward(nn.Module):
  """Positional-wise feed forward network."""

  def __init__(self, model_dim=512, ffn_dim=2048, dropout=0.0):
    """Init.
    Args:
      model_dim: Model's dimension, default is 512 according to the paper.
      ffn_dim: Hidden size of the feed forward network,
        default is 2048 according to the paper.
      dropout: Dropout rate.
    """
    super(PositionalWiseFeedForward, self).__init__()
    self.w1 = nn.Conv1d(model_dim, ffn_dim, 1)
    self.w2 = nn.Conv1d(model_dim, ffn_dim, 1)
    self.dropout = nn.Dropout(dropout)
    self.layer_norm = nn.LayerNorm(model_dim)

  def forward(self, x):
    """Forward pass.
    Args:
      x: Input tensor, with shape of [B, L, D]
    Returns:
      A tensor with shape of [B, L, D], without residual value and normalization
    """
    output = x.transpose(1, 2)
    output = self.w2(F.relu(self.w1(output)))
    output = self.dropout(output.transpose(1, 2))

    # add residual and norm layer
    output = self.layer_norm(x + output)
    return output


class EncoderLayer(nn.Module):
  """A encoder block, with tow sub layers."""

  def __init__(self, model_dim=512, num_heads=8, ffn_dim=2018, dropout=0.0):
    super(EncoderLayer, self).__init__()

    self.attention = MultiHeadAttention(model_dim, num_heads, dropout)
    self.feed_forward = PositionalWiseFeedForward(model_dim, ffn_dim, dropout)

  def forward(self, inputs, attn_mask=None):
    """Forward pass.
    Args:
      inputs: Embedded inputs tensor, with shape [B, L, D]
      attn_mask: Binary mask tensor for attention, with shape [B, L, L]
    Returns:
      Output and attention tensors of encoder layer.
    """

    # self attention
    context, attention = self.attention(inputs, inputs, inputs, padding_mask)

    # feed forward network
    output = self.feed_forward(context)

    return output, attention


class Encoder(nn.Module):

  def __init__(self,
               vocab_size,
               max_seq_len,
               num_layers=6,
               model_dim=512,
               num_heads=8,
               ffn_dim=2048,
               dropout=0.0):
    super(Encoder, self).__init__()

    self.encoder_layers = nn.ModuleList(
      [EncoderLayer(model_dim, num_heads, ffn_dim, dropout) for _ in
       range(num_layers)])

    self.seq_embedding = nn.Embedding(vocab_size + 1, model_dim, padding_idx=0)
    self.pos_embedding = PositionalEncoding(model_dim, max_seq_len)

  def forward(self, inputs, inputs_len):
    """Forward pass.
    Args:
      inputs: embedded inputs
      inputs_len: length of input sequence
    Returns:
      An output of encoder block.
      An attention list contains each attention tensor of each encoder layer.
    """
    output = self.seq_embedding(inputs)
    output += self.pos_embedding(inputs_len)

    self_attention_mask = padding_mask(inputs, inputs)

    attentions = []
    for encoder in self.encoder_layers:
      output, attention = encoder(output, self_attention_mask)
      attentions.append(attention)

    return output, attentions


class DecoderLayer(nn.Module):

  def __init__(self, model_dim, num_heads=8, ffn_dim=2048, dropout=0.0):
    super(DecoderLayer, self).__init__()

    self.attention = MultiHeadAttention(model_dim, num_heads, dropout)
    self.feed_forward = PositionalWiseFeedForward(model_dim, ffn_dim, dropout)

  def forward(self,
              dec_inputs,
              enc_outputs,
              self_attn_mask=None,
              context_attn_mask=None):
    """Forward pass.
    Args:
      dec_inputs: Embedded input tensor
      enc_outputs: Encoder's output
      self_attn_mask: Mask tensor, with shape of [B, L, L], pad_mask + seq_mask
      context_attn_mask: Padding mask tensor, with shape of [B, L_q, L_k]
    """

    # self attention, all inputs are decoder inputs
    dec_output, self_attention = self.attention(
      dec_inputs, dec_inputs, dec_inputs, self_attn_mask)

    # context attention
    # query is decoder's outputs, key and value are encoder's inputs
    dec_output, context_attention = self.attention(
      enc_outputs, enc_outputs, dec_output, context_attn_mask)

    # decoder's output, or context
    dec_output = self.feed_forward(dec_output)

    return dec_output, self_attention, context_attention


class Decoder(nn.Module):

  def __init__(self,
               vocab_size,
               max_seq_len,
               num_layers=6,
               model_dim=512,
               num_heads=8,
               ffn_dim=2048,
               dropout=0.0):
    super(Decoder, self).__init__()

    self.num_layers = num_layers

    self.decoder_layers = nn.ModuleList(
      [DecoderLayer(model_dim, num_heads, ffn_dim, dropout) for _ in
       range(num_layers)])

    self.seq_embedding = nn.Embedding(vocab_size + 1, model_dim, padding_idx=0)
    self.pos_embedding = PositionalEncoding(model_dim, max_seq_len)

  def forward(self, inputs, inputs_len, enc_output, context_attn_mask=None):
    """Forward pass.
    Args:
      inputs: Embedded inputs
      inputs_len: Length tensor of inputs
      enc_output: Encoder's output
      context_attn_mask: Mask tensor for context attention
    """
    output = self.seq_embedding(inputs)
    output += self.pos_embedding(inputs_len)

    self_attention_padding_mask = padding_mask(inputs, inputs)
    seq_mask = sequence_mask(inputs)
    self_attn_mask = torch.gt((self_attention_padding_mask + seq_mask), 0)

    self_attentions = []
    context_attentions = []
    for decoder in self.decoder_layers:
      output, self_attn, context_attn = decoder(
        output, enc_output, self_attn_mask, context_attn_mask)
      self_attentions.append(self_attn)
      context_attentions.append(context_attn)

    return output, self_attentions, context_attentions


def sequence_mask(seq):
  """Sequence mask to masking out sub-sequence info.
  Args:
    seq: Sequence tensor, with shape [B, L]
  Returns:
    A masking tensor, with shape [B, L, L]
  """
  batch_size, seq_len = seq.size()
  mask = torch.triu(torch.ones((seq_len, seq_len), dtype=torch.uint8),
                    diagonal=1)
  mask = mask.unsqueeze(0).expand(batch_size, -1, -1)  # [B, L, L]
  return mask


def padding_mask(seq_k, seq_q):
  """For masking out the padding part of the keys sequence.
  Args:
    seq_k: Keys tensor, with shape [B, L_k]
    seq_q: Query tensor, with shape [B, L_q]
  Returns:
    A masking tensor, with shape [B, L_1, L_k]
  """
  len_q = seq_q.size(1)
  # `PAD` is 0
  pad_mask = seq_k.eq(0)
  pad_mask = pad_mask.unsqueeze(1).expand(-1, len_q, -1)  # shape [B, L_q, L_k]
  return pad_mask


class Transformer(nn.Module):

  def __init__(self,
               src_vocab_size,
               src_max_len,
               tgt_vocab_size,
               tgt_max_len,
               num_layers=6,
               model_dim=512,
               num_heads=8,
               ffn_dim=2048,
               dropout=0.2):
    super(Transformer, self).__init__()

    self.encoder = Encoder(src_vocab_size, src_max_len, num_layers, model_dim,
                           num_heads, ffn_dim, dropout)
    self.decoder = Decoder(tgt_vocab_size, tgt_max_len, num_layers, model_dim,
                           num_heads, ffn_dim, dropout)

    self.linear = nn.Linear(model_dim, tgt_vocab_size, bias=False)
    self.softmax = nn.Softmax(dim=2)

  def forward(self, src_seq, src_len, tgt_seq, tgt_len):
    context_attn_mask = padding_mask(tgt_seq, src_seq)

    output, enc_self_attn = self.encoder(src_seq, src_len)

    output, dec_self_attn, ctx_attn = self.decoder(
      tgt_seq, tgt_len, output, context_attn_mask)

    output = self.linear(output)
    output = self.softmax(output)

    return output, enc_self_attn, dec_self_attn, ctx_attn


class Trans(nn.Module):
    def __init__(self, n_dims):
        super(Trans, self).__init__()

        self.query = nn.Conv1d(n_dims, n_dims, kernel_size=1)
        self.key = nn.Conv1d(n_dims, n_dims, kernel_size=1)
        self.value = nn.Conv1d(n_dims, n_dims, kernel_size=1)
        # # nn.Parameter 含义是将一个固定不可训练的tensor转换成可以训练的类型parameter
        # self.rel_h = nn.Parameter(torch.randn([1, n_dims, 1, height]), requires_grad=True)
        # self.rel_w = nn.Parameter(torch.randn([1, n_dims, width, 1]), requires_grad=True)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        n_batch, in_channels = x.size()
        q = self.query(x).view(n_batch, in_channels, -1)
        k = self.key(x).view(n_batch, in_channels, -1)
        v = self.value(x).view(n_batch, in_channels, -1)
        # 对存储在两个批bach1和batch内的矩阵进行批矩阵乘操作。batch1和2都包含相同数量矩阵的三维张量
        content_content = torch.bmm(q.permute(0, 2, 1), k)

        # content_position = (self.rel_h + self.rel_w).view(1, C, -1).permute(0, 2, 1)
        content_position = torch.matmul(q)

        energy = content_content + content_position
        attention = self.softmax(energy)

        out = torch.bmm(v, attention.permute(0, 2, 1))
        out = out.view(n_batch, in_channels)

        return out


class CNN(nn.Module):
    def __init__(self, num_filters, k_size):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=num_filters*2, kernel_size=k_size, stride=1, padding=k_size//2),
            
        )
        self.conv2 = nn.Sequential(
            nn.Dropout(0.2),
            nn.Conv1d(num_filters, num_filters * 4, k_size, 1, k_size//2),
            
        )
        self.conv3 = nn.Sequential(
            nn.Dropout(0.2),
            nn.Conv1d(num_filters * 2, num_filters * 6, k_size, 1, k_size//2),
            
        )
        self.conv4 = Trans(num_filters)

        self.out = nn.AdaptiveAvgPool1d(1)
        self.layer1 = nn.Sequential(
            nn.Linear(num_filters * 3, num_filters * 3),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Linear(num_filters * 3, num_filters * 3),
            nn.ReLU()
        )

    def reparametrize(self, mean, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.cuda.FloatTensor(std.size()).normal_(0,0.1)
        eps = Variable(eps)
        return eps.mul(std).add_(mean)

    def forward(self, x):
        x = self.conv1(x)
        out, gate = x.split(int(x.size(1) / 2), 1)
        x = out * torch.sigmoid(gate)
        x = self.conv2(x)
        out, gate = x.split(int(x.size(1) / 2), 1)
        x = out * torch.sigmoid(gate)
        x = self.conv3(x)
        out, gate = x.split(int(x.size(1) / 2), 1)
        x = out * torch.sigmoid(gate)
        output = self.out(x)
        output = output.squeeze()
        output1 = self.layer1(output)
        output2 = self.layer2(output)
        output = self.reparametrize(output1, output2)
        return output, output1, output2


class decoder(nn.Module):
    def __init__(self, init_dim, num_filters, k_size,size):
        super(decoder, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(num_filters * 3, num_filters * 3 * (init_dim - 3 * (k_size - 1))),
            nn.ReLU()
        )
        self.convt = nn.Sequential(
            nn.ConvTranspose1d(num_filters * 3, num_filters * 2, k_size, 1, 0),
            nn.ReLU(),
            nn.ConvTranspose1d(num_filters * 2, num_filters, k_size, 1, 0),
            nn.ReLU(),
            nn.ConvTranspose1d(num_filters, 128, k_size, 1, 0),
            nn.ReLU(),
        )
        self.layer2 = nn.Linear(128, size)

    def forward(self, x, init_dim, num_filters, k_size):
        # print(x.shape)
        x = self.layer(x)
        x = x.view(-1, num_filters * 3, init_dim - 3 * (k_size - 1))
        # print(x.shape)
        x = self.convt(x)
        x = x.permute(0,2,1)
        x = self.layer2(x)
        return x


class net_reg(nn.Module):
    def __init__(self, num_filters):
        super(net_reg, self).__init__()
        self.reg = nn.Sequential(
            nn.Linear(num_filters * 6, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 1)
        )

        self.reg1 = nn.Sequential(
            nn.Linear(num_filters * 3, num_filters * 3),
            nn.ReLU()
        )

        self.reg2 = nn.Sequential(
            nn.Linear(num_filters * 3, num_filters * 3),
            nn.ReLU()
        )


    def forward(self, A, B):
        A = self.reg1(A)
        B = self.reg2(B)
        x = torch.cat((A, B), 1)
        x = self.reg(x)
        return x

class net(nn.Module):
    def __init__(self, FLAGS , NUM_FILTERS , FILTER_LENGTH1 , FILTER_LENGTH2):
        super(net, self).__init__()
        self.embedding1 = nn.Embedding(FLAGS.charsmiset_size, 128)
        self.embedding2 = nn.Embedding(FLAGS.charseqset_size, 128)
        self.cnn1 = CNN(NUM_FILTERS, FILTER_LENGTH1)
        self.cnn2 = CNN(NUM_FILTERS, FILTER_LENGTH2)
        self.reg = net_reg(NUM_FILTERS)
        self.decoder1 = decoder(FLAGS.max_smi_len, NUM_FILTERS, FILTER_LENGTH1,FLAGS.charsmiset_size)
        self.decoder2 = decoder(FLAGS.max_seq_len, NUM_FILTERS, FILTER_LENGTH2,FLAGS.charseqset_size)

    def forward(self, x, y, FLAGS, NUM_FILTERS, FILTER_LENGTH1, FILTER_LENGTH2):
        x_init = Variable(x.long()).cuda()
        x = self.embedding1(x_init)
        x_embedding = x.permute(0, 2, 1)
        y_init = Variable(y.long()).cuda()
        y = self.embedding2(y_init)
        y_embedding = y.permute(0, 2, 1)
        # print('______________________________________x_embedding___________________________________')
        # print(x_embedding.shape)
        # print('______________________________________y_embedding___________________________________')
        # print(y_embedding.shape)
        x, mu_x, logvar_x = self.cnn1(x_embedding)
        # print('______________________________________x,mu_x,logvar_x___________________________________')
        # print(x.shape)
        # print(mu_x.shape)
        # print(logvar_x.shape)
        y, mu_y, logvar_y = self.cnn2(y_embedding)
        # print('______________________________________y,mu_y,logvar_y___________________________________')
        # print(y.shape)
        # print(mu_y.shape)
        # print(logvar_y.shape)
        out = self.reg(x, y).squeeze()
        x = self.decoder1(x, FLAGS.max_smi_len, NUM_FILTERS, FILTER_LENGTH1)
        y = self.decoder2(y, FLAGS.max_seq_len, NUM_FILTERS, FILTER_LENGTH2)
        return out, x, y, x_init, y_init, mu_x, logvar_x, mu_y, logvar_y

# if __name__ == "__main__":
#
#     net()
#     print('111')




