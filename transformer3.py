import torch
from torch import nn
import numpy as np
from scipy.stats import norm
device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
channel_number = 512
num_pixels =196

class PoswiseFeedForwardNet2(nn.Module):
    def __init__(self, embed_dim, d_ff, dropout):
        super(PoswiseFeedForwardNet2, self).__init__()
        """
        Two fc layers can also be described by two cnn with kernel_size=1.
        """
        self.conv1 = nn.Conv1d(in_channels=2*embed_dim, out_channels=d_ff, kernel_size=1).to(device)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=embed_dim, kernel_size=1).to(device)
        self.dropout = nn.Dropout(p=dropout)
        self.embed_dim = embed_dim

    def forward(self, inputs_x,inputs_y):
        """
        encoder: inputs: [batch_size, len_q=196, embed_dim=2048]
        decoder: inputs: [batch_size, max_len=52, embed_dim=512]
        """
        residual = inputs_x + inputs_y
        output = nn.Sigmoid()(self.conv1(torch.cat([inputs_x,inputs_y],dim=2).transpose(1, 2)))
        output = self.conv2(output).transpose(1, 2)
        output = self.dropout(output)
        return nn.LayerNorm(self.embed_dim).to(device)(output + residual ) #


class ScaledDotProductAttention(nn.Module):
    def __init__(self, QKVdim):
        super(ScaledDotProductAttention, self).__init__()
        self.QKVdim = QKVdim

    def forward(self, Q, K, V, attn_mask):
        """
        :param Q: [batch_size, n_heads, -1(len_q), QKVdim]
        :param K, V: [batch_size, n_heads, -1(len_k=len_v), QKVdim]
        :param attn_mask: [batch_size, n_heads, len_q, len_k]
        """
        # scores: [batch_size, n_heads, len_q, len_k]
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.QKVdim)
        # Fills elements of self tensor with value where mask is True.
        scores.to(device).masked_fill_(attn_mask, -1e9)
        attn = nn.Softmax(dim=-1)(scores)  # [batch_size, n_heads, len_q, len_k]
        context = torch.matmul(attn, V).to(device)  # [batch_size, n_heads, len_q, QKVdim]
        return context, attn


class Multi_Head_Attention(nn.Module):
    def __init__(self, Q_dim, K_dim, QKVdim, n_heads=8, dropout=0.1):
        super(Multi_Head_Attention, self).__init__()
        self.W_Q = nn.Linear(Q_dim, QKVdim * n_heads).to(device)
        self.W_K = nn.Linear(K_dim, QKVdim * n_heads).to(device)
        self.W_V = nn.Linear(K_dim, QKVdim * n_heads).to(device)
        self.n_heads = n_heads
        self.QKVdim = QKVdim
        self.embed_dim = Q_dim
        self.dropout = nn.Dropout(p=dropout)
        self.W_O = nn.Linear(self.n_heads * self.QKVdim, self.embed_dim).to(device)

    def forward(self, Q, K, V, attn_mask):
        """
        In self-encoder attention:
                Q = K = V: [batch_size, num_pixels=196, encoder_dim=2048]
                attn_mask: [batch_size, len_q=196, len_k=196]
        In self-decoder attention:
                Q = K = V: [batch_size, max_len=52, embed_dim=512]
                attn_mask: [batch_size, len_q=52, len_k=52]
        encoder-decoder attention:
                Q: [batch_size, 52, 512] from decoder
                K, V: [batch_size, 196, 2048] from encoder
                attn_mask: [batch_size, len_q=52, len_k=196]
        return _, attn: [batch_size, n_heads, len_q, len_k]
        """
        residual, batch_size = Q, Q.size(0)
        # q_s: [batch_size, n_heads=8, len_q, QKVdim] k_s/v_s: [batch_size, n_heads=8, len_k, QKVdim]
        q_s = self.W_Q(Q).view(batch_size, -1, self.n_heads, self.QKVdim).transpose(1, 2)
        k_s = self.W_K(K).view(batch_size, -1, self.n_heads, self.QKVdim).transpose(1, 2)
        v_s = self.W_V(V).view(batch_size, -1, self.n_heads, self.QKVdim).transpose(1, 2)
        # attn_mask: [batch_size, self.n_heads, len_q, len_k]
        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)
        # attn: [batch_size, n_heads, len_q, len_k]
        # context: [batch_size, n_heads, len_q, QKVdim]
        context, attn = ScaledDotProductAttention(self.QKVdim)(q_s, k_s, v_s, attn_mask)
        # context: [batch_size, n_heads, len_q, QKVdim] -> [batch_size, len_q, n_heads * QKVdim]
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.QKVdim).to(device)
        # output: [batch_size, len_q, embed_dim]
        output = self.W_O(context)
        output = self.dropout(output)
        return nn.LayerNorm(self.embed_dim).to(device)(output + residual), attn


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, embed_dim, d_ff, dropout):
        super(PoswiseFeedForwardNet, self).__init__()
        """
        Two fc layers can also be described by two cnn with kernel_size=1.
        """
        self.conv1 = nn.Conv1d(in_channels=embed_dim, out_channels=d_ff, kernel_size=1).to(device)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=embed_dim, kernel_size=1).to(device)
        self.dropout = nn.Dropout(p=dropout)
        self.embed_dim = embed_dim

    def forward(self, inputs):
        """
        encoder: inputs: [batch_size, len_q=196, embed_dim=2048]
        decoder: inputs: [batch_size, max_len=52, embed_dim=512]
        """
        residual = inputs
        output = nn.ReLU()(self.conv1(inputs.transpose(1, 2)))
        output = self.conv2(output).transpose(1, 2)
        output = self.dropout(output)
        return nn.LayerNorm(self.embed_dim).to(device)(output + residual)


class DecoderLayer(nn.Module):
    def __init__(self, embed_dim,encoder_out_dim, dropout, attention_method, n_heads):
        super(DecoderLayer, self).__init__()
        self.dec_self_attn = Multi_Head_Attention(Q_dim=embed_dim, K_dim=embed_dim, QKVdim=64, n_heads=n_heads, dropout=dropout)
        if attention_method == "ByPixel":
            self.dec_enc_attn = Multi_Head_Attention(Q_dim=embed_dim, K_dim=encoder_out_dim, QKVdim=64, n_heads=n_heads, dropout=dropout)
            self.pos_ffn = PoswiseFeedForwardNet(embed_dim=embed_dim, d_ff=encoder_out_dim, dropout=dropout)
        elif attention_method == "ByChannel":
            self.dec_enc_attn = Multi_Head_Attention(Q_dim=embed_dim, K_dim=196, QKVdim=64, n_heads=n_heads, dropout=dropout)
            self.pos_ffn = PoswiseFeedForwardNet(embed_dim=embed_dim, d_ff=2048, dropout=dropout)  # need to change

    def forward(self, dec_inputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask):
        """
        :param dec_inputs: [batch_size, max_len=52, embed_dim=512]
        :param enc_outputs: [batch_size, num_pixels=196, 2048]
        :param dec_self_attn_mask: [batch_size, 52, 52]
        :param dec_enc_attn_mask: [batch_size, 52, 196]
        """
        dec_outputs, dec_self_attn = self.dec_self_attn(dec_inputs, dec_inputs, dec_inputs, dec_self_attn_mask)
        dec_outputs, dec_enc_attn = self.dec_enc_attn(dec_outputs, enc_outputs, enc_outputs, dec_enc_attn_mask)
        dec_outputs = self.pos_ffn(dec_outputs)
        return dec_outputs, dec_self_attn, dec_enc_attn


class Decoder(nn.Module):
    def __init__(self, n_layers, vocab_size, embed_dim,encoder_out_dim, dropout, attention_method, n_heads):
        super(Decoder, self).__init__()
        self.vocab_size = vocab_size
        self.tgt_emb = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_emb = nn.Embedding.from_pretrained(self.get_position_embedding_table(embed_dim), freeze=True)
        self.dropout = nn.Dropout(p=dropout)
        self.layers = nn.ModuleList([DecoderLayer(embed_dim, encoder_out_dim, dropout, attention_method, n_heads) for _ in range(n_layers)])
        self.projection = nn.Linear(embed_dim, vocab_size, bias=False).to(device)
        self.attention_method = attention_method

        self.decoder_aggregation = [0] * (n_layers - 1)
        for i in range(n_layers - 1):
            self.decoder_aggregation[i] = PoswiseFeedForwardNet2(embed_dim=embed_dim, d_ff=encoder_out_dim, dropout=dropout)

    def get_position_embedding_table(self, embed_dim):
        def cal_angle(position, hid_idx):
            return position / np.power(10000, 2 * (hid_idx // 2) / embed_dim)
        def get_posi_angle_vec(position):
            return [cal_angle(position, hid_idx) for hid_idx in range(embed_dim)]

        embedding_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(52)])
        embedding_table[:, 0::2] = np.sin(embedding_table[:, 0::2])  # dim 2i
        embedding_table[:, 1::2] = np.cos(embedding_table[:, 1::2])  # dim 2i+1
        return torch.FloatTensor(embedding_table).to(device)

    def get_attn_pad_mask(self, seq_q, seq_k):
        batch_size, len_q = seq_q.size()
        batch_size, len_k = seq_k.size()
        # In wordmap, <pad>:0
        # pad_attn_mask: [batch_size, 1, len_k], one is masking
        pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)
        return pad_attn_mask.expand(batch_size, len_q, len_k)  # [batch_size, len_q, len_k]

    def get_attn_subsequent_mask(self, seq):
        attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
        subsequent_mask = np.triu(np.ones(attn_shape), k=1)
        subsequent_mask = torch.from_numpy(subsequent_mask).byte().to(device)
        return subsequent_mask

    def forward(self, encoder_out, encoded_captions, caption_lengths):
        """
        :param encoder_out: [batch_size, num_pixels=196, 2048]
        :param encoded_captions: [batch_size, 52]
        :param caption_lengths: [batch_size, 1]
        """
        batch_size = encoder_out.size(0)
        # Sort input data by decreasing lengths.
        caption_lengths, sort_ind = caption_lengths.squeeze(1).sort(dim=0, descending=True)
        encoder_out = encoder_out[sort_ind]
        encoded_captions = encoded_captions[sort_ind]
        # We won't decode at the <end> position, since we've finished generating as soon as we generate <end>
        # So, decoding lengths are actual lengths - 1
        decode_lengths = (caption_lengths - 1).tolist()

        # dec_outputs: [batch_size, max_len=52, embed_dim=512]
        # dec_self_attn_pad_mask: [batch_size, len_q=52, len_k=52], 1 if id=0(<pad>)
        # dec_self_attn_subsequent_mask: [batch_size, 52, 52], Upper triangle of an array with 1.
        # dec_self_attn_mask for self-decoder attention, the position whose val > 0 will be masked.
        # dec_enc_attn_mask for encoder-decoder attention.
        # e.g. 9488, 23, 53, 74, 0, 0  |  dec_self_attn_mask:
        # 0 1 1 1 2 2
        # 0 0 1 1 2 2
        # 0 0 0 1 2 2
        # 0 0 0 0 2 2
        # 0 0 0 0 1 2
        # 0 0 0 0 1 1
        dec_outputs = self.tgt_emb(encoded_captions) + self.pos_emb(torch.LongTensor([list(range(52))]*batch_size).to(device))
        dec_outputs = self.dropout(dec_outputs)
        dec_self_attn_pad_mask = self.get_attn_pad_mask(encoded_captions, encoded_captions)
        dec_self_attn_subsequent_mask = self.get_attn_subsequent_mask(encoded_captions)
        dec_self_attn_mask = torch.gt((dec_self_attn_pad_mask + dec_self_attn_subsequent_mask), 0)
        if self.attention_method == "ByPixel":
            dec_enc_attn_mask = (torch.tensor(np.zeros((batch_size, 52, num_pixels))).to(device) == torch.tensor(np.ones((batch_size, 52, num_pixels))).to(device))
        elif self.attention_method == "ByChannel":
            dec_enc_attn_mask = (torch.tensor(np.zeros((batch_size, 52, channel_number))).to(device) == torch.tensor(np.ones((batch_size, 52, channel_number))).to(device))

        dec_self_attns, dec_enc_attns = [], []
        decoderlayers_outputs = []
        for layer in self.layers:
            # attn: [batch_size, n_heads, len_q, len_k]
            dec_outputs, dec_self_attn, dec_enc_attn = layer(dec_outputs, encoder_out, dec_self_attn_mask, dec_enc_attn_mask)
            dec_self_attns.append(dec_self_attn)
            dec_enc_attns.append(dec_enc_attn)
            decoderlayers_outputs.append(dec_outputs)
        # # FIXME:
        # dec_outputs = decoderlayers_outputs[0]
        # for i in range(len(decoderlayers_outputs)-1):
        #     dec_outputs = self.decoder_aggregation[i](decoderlayers_outputs[i + 1], dec_outputs)
        predictions = self.projection(dec_outputs)
        return predictions, encoded_captions, decode_lengths, sort_ind, dec_self_attns, dec_enc_attns


class EncoderLayer(nn.Module):
    def __init__(self, dropout, attention_method, n_heads):
        super(EncoderLayer, self).__init__()
        """
        In "Attention is all you need" paper, dk = dv = 64, h = 8, N=6
        """
        if attention_method == "ByPixel":
            self.enc_self_attn = Multi_Head_Attention(Q_dim=2048, K_dim=2048, QKVdim=64, n_heads=n_heads, dropout=dropout)
            self.pos_ffn = PoswiseFeedForwardNet(embed_dim=2048, d_ff=4096, dropout=dropout)
        elif attention_method == "ByChannel":
            self.enc_self_attn = Multi_Head_Attention(Q_dim=196, K_dim=196, QKVdim=64, n_heads=n_heads, dropout=dropout)
            self.pos_ffn = PoswiseFeedForwardNet(embed_dim=196, d_ff=512, dropout=dropout)

    def forward(self, enc_inputs, enc_self_attn_mask):
        """
        :param enc_inputs: [batch_size, num_pixels=196, 2048]
        :param enc_outputs: [batch_size, len_q=196, d_model=2048]
        :return: attn: [batch_size, n_heads=8, 196, 196]
        """
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask)
        enc_outputs = self.pos_ffn(enc_outputs)
        return enc_outputs, attn


class Encoder(nn.Module):
    def __init__(self, encoder_input_dim, n_layers, dropout, attention_method, n_heads):
        super(Encoder, self).__init__()
        self.encoder_input_dim = encoder_input_dim
        if attention_method == "ByPixel":
            self.pos_emb = nn.Embedding.from_pretrained(self.get_position_embedding_table(), freeze=True)
        # self.dropout = nn.Dropout(p=dropout)
        self.layers = nn.ModuleList([EncoderLayer(dropout, attention_method, n_heads) for _ in range(n_layers)])
        self.attention_method = attention_method

    def get_position_embedding_table(self):

        # 1、原始GitHub，考虑二位信息但是写的不对，不能是sin(x)+sin(y);而且没考虑奇数和偶数通道区别
        # def cal_angle(position, hid_idx):
        #     x = position % 14
        #     y = position // 14
        #     x_enc = x / np.power(10000, 2*hid_idx / self.encoder_input_dim)   # FIXME:令dmodel=self.encoder_input_dim才能进行encoder_out = encoder_out + PE
        #     y_enc = y / np.power(10000, 2*hid_idx / self.encoder_input_dim)
        #     return np.sin(x_enc), np.sin(y_enc)
        # def get_posi_angle_vec(position):
        #     return [cal_angle(position, hid_idx)[0] for hid_idx in range(int(self.encoder_input_dim/2))] + [cal_angle(position, hid_idx)[1] for hid_idx in range(int(self.encoder_input_dim/2))]

        # 2、transformer论文里的不考虑二维信息，考虑了奇数和偶数通道区别
        def cal_angle(position, hid_idx):
            x_enc = position / np.power(10000, 2*hid_idx / 2048)   # dmodel=2048
            if hid_idx % 2 ==0:
                return np.sin(x_enc)
            else:
                return np.cos(x_enc)
        def get_posi_angle_vec(position):
            return [cal_angle(position, hid_idx) for hid_idx in range(2048)]

        # 3、FIXME:my PE,考虑了二维信息以及奇数和偶数通道区别
        # def cal_angle(position, hid_idx):
        #     x = position % 14
        #     y = position // 14
        #     x_enc = x / np.power(10000, hid_idx / 1024)
        #     y_enc = y / np.power(10000, hid_idx / 1024)
        #     if hid_idx % 2 ==0:
        #         return np.sin(x_enc), np.cos(y_enc)
        #     else:
        #         return np.cos(x_enc), np.sin(y_enc)
        # def get_posi_angle_vec(position):
        #     return [cal_angle(position, hid_idx)[0] for hid_idx in range(1024)] + [cal_angle(position, hid_idx)[1] for hid_idx in range(1024)]

        # # 4、FIXME:高斯
        # def cal_angle(position, hid_idx):
        #     x = position % 14
        #     y = position // 14
        #     x_enc = x / np.power(10000, hid_idx / 1024)
        #     y_enc = y / np.power(10000, hid_idx / 1024)
        #     mu = 6.5
        #     sigma = 0.1
        #     x_enc2 = norm.pdf(x, mu, sigma)
        #     y_enc2 = norm.pdf(y, mu, sigma)
        #     return np.sin(x_enc), np.sin(y_enc), x_enc2, y_enc2
        # def get_posi_angle_vec(position):
        #     return [cal_angle(position, hid_idx)[0] for hid_idx in range(512)] + [cal_angle(position, hid_idx)[1]for hid_idx in range(512)] + [cal_angle(position, hid_idx)[2]for hid_idx in range(512)]+ [cal_angle(position, hid_idx)[3]for hid_idx in range(512)]

        # 5.FIXME：x,y错位融合
        # def cal_angle(position, hid_idx):
        #     x = position % 14
        #     y = position // 14
        #     x_enc = x / np.power(10000, hid_idx / 1024)
        #     y_enc = y / np.power(10000, hid_idx / 1024)
        #     return np.sin(x_enc), np.sin(y_enc)
        # def get_posi_angle_vec(position):
        #     return [cal_angle(position, hid_idx)[0] for hid_idx in range(1024)] + [cal_angle(position, hid_idx)[1] for hid_idx in range(1024)]

        embedding_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(num_pixels)])
        return torch.FloatTensor(embedding_table).to(device)



    def forward(self, encoder_out):
        """
        :param encoder_out: [batch_size, num_pixels=196, dmodel=2048]
        """
        batch_size = encoder_out.size(0)
        positions = encoder_out.size(1)
        if self.attention_method == "ByPixel":
            encoder_out = encoder_out + self.pos_emb(torch.LongTensor([list(range(positions))]*batch_size).to(device))
        # encoder_out = self.dropout(encoder_out)
        # enc_self_attn_mask: [batch_size, 196, 196]
        enc_self_attn_mask = (torch.tensor(np.zeros((batch_size, positions, positions))).to(device)
                              == torch.tensor(np.ones((batch_size, positions, positions))).to(device))
        enc_self_attns = []
        encoderlayers_out = []
        for layer in self.layers:
            encoder_out, enc_self_attn = layer(encoder_out, enc_self_attn_mask)
            encoderlayers_out.append(encoder_out)
            # encoder_out = encoder_out + self.pos_emb(torch.LongTensor([list(range(positions))] * batch_size).to(device))  # FIXME:
            enc_self_attns.append(enc_self_attn)
        return encoder_out, encoderlayers_out,enc_self_attns


class Transformer(nn.Module):
    """
    See paper 5.4: "Attention Is All You Need" - https://arxiv.org/abs/1706.03762
    "Apply dropout to the output of each sub-layer, before it is added to the sub-layer input and normalized.
    In addition, apply dropout to the sums of the embeddings and the positional encodings in both the encoder
    and decoder stacks." (Now, we dont't apply dropout to the encoder embeddings)
    """
    def __init__(self, vocab_size, embed_dim, encoder_layers, decoder_layers, dropout=0.1, attention_method="ByPixel", n_heads=8):
        super(Transformer, self).__init__()
        encoder_input_dim =2048
        self.encoder = Encoder(encoder_input_dim, encoder_layers, dropout, attention_method, n_heads)

        encoder_out_dim = encoder_input_dim   # 4096
        self.decoder = Decoder(decoder_layers, vocab_size, embed_dim, encoder_out_dim, dropout, attention_method, n_heads)
        self.embedding = self.decoder.tgt_emb
        self.attention_method = attention_method
        # FIXME:fangshi1
        # self.w1 = nn.Linear(in_features = 2048, out_features = 2048)
        # self.w2 = nn.Linear(in_features=2048, out_features=2048)
        # self.w3 = nn.Linear(in_features=2048, out_features=2048)
        # FIXME:fangshi2
        # self.encoder_aggregation = [0]*(encoder_layers-1)
        # for i in range(encoder_layers-1):
        #     self.encoder_aggregation[i] = PoswiseFeedForwardNet2(embed_dim=encoder_input_dim, d_ff=4096, dropout=dropout)

        # FIXME:fangshi4
        self.conv_1 = nn.Conv2d(in_channels=2048, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv_2 = nn.Conv2d(in_channels=2048, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv_3 = nn.Conv2d(in_channels=2048, out_channels=1024, kernel_size=3, stride=1, padding=1)
        self.conv_4 = nn.Conv2d(in_channels=2048, out_channels=2048, kernel_size=3, stride=1, padding=1)
        # self.conv_5 = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=1, stride=1)
        # self.conv_5 = nn.Sequential(
        #     nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1),
        #     nn.ReLU(),
        #     nn.Dropout(0.5),
        #     nn.Conv2d(in_channels=16, out_channels=1, kernel_size=3, stride=1, padding=1),
        #     nn.ReLU(),
        #     nn.Dropout(0.5)
        # )
        self.conv_5 = nn.Sequential(
            nn.Conv2d(in_channels=6144, out_channels=2048, kernel_size=3,stride=1, padding=1),#
            nn.BatchNorm2d(2048),
            nn.ReLU(),
            nn.Dropout(0.5)
            # nn.Conv2d(in_channels=2048, out_channels=2048, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(2048),
            # nn.ReLU(),
            # nn.Dropout(0.5)
        )

        # FIXME:下面是初始化LSTM的
        # self.lstm_embedding = nn.Embedding(encoder_input_dim, encoder_input_dim)  # embedding layer
        # self.dropout = nn.Dropout(p=dropout)
        # self.decode_step = nn.LSTMCell(encoder_input_dim, encoder_input_dim, bias=True)  # decoding LSTMCell，只在初始时刻输入特征
        # self.init_h = nn.Linear(encoder_input_dim, encoder_input_dim)  # linear layer to find initial hidden state of LSTMCell
        # self.init_c = nn.Linear(encoder_input_dim, encoder_input_dim)  # linear layer to find initial cell state of LSTMCell
        # self.f_beta = nn.Linear(encoder_input_dim, encoder_input_dim)  # linear layer to create a sigmoid-activated gate
        # self.sigmoid = nn.Sigmoid()
        # self.fc = nn.Linear(encoder_input_dim, encoder_input_dim)  # linear layer to find scores over vocabulary
        # self.init_weights()  # initialize some layers with the uniform distribution

    def load_pretrained_embeddings(self, embeddings):
        self.embedding.weight = nn.Parameter(embeddings)

    def fine_tune_embeddings(self, fine_tune=True):
        for p in self.embedding.parameters():
            p.requires_grad = fine_tune

    # FIXME:下面是LSTM相关的两个函数
    def init_weights(self):
        """
        Initializes some parameters with values from the uniform distribution, for easier convergence.
        """
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def init_hidden_state(self, encoder_out):
        """
        Creates the initial hidden and cell states for the decoder's LSTM based on the encoded images.

        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :return: hidden state, cell state
        """
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)  # (batch_size, decoder_dim)
        c = self.init_c(mean_encoder_out)
        return h, c


    def forward(self, enc_inputs, encoded_captions, caption_lengths):
        """
        preprocess: enc_inputs: [batch_size, 14, 14, 2048]/[batch_size, 196, 2048] -> [batch_size, 196, 2048]
        encoded_captions: [batch_size, 52]
        caption_lengths: [batch_size, 1], not used
        The encoder or decoder is composed of a stack of n_layers=6 identical layers.
        One layer in encoder: Multi-head Attention(self-encoder attention) with Norm & Residual
                            + Feed Forward with Norm & Residual
        One layer in decoder: Masked Multi-head Attention(self-decoder attention) with Norm & Residual
                            + Multi-head Attention(encoder-decoder attention) with Norm & Residual
                            + Feed Forward with Norm & Residual
        """
        batch_size = enc_inputs.size(0)
        encoder_dim = enc_inputs.size(-1)

        if self.attention_method == "ByPixel":
            enc_inputs = enc_inputs.view(batch_size, -1, encoder_dim)
        # elif self.attention_method == "ByChannel":
        #     enc_inputs = enc_inputs.view(batch_size, -1, encoder_dim).permute(0, 2, 1)  # (batch_size, 2048, 196)

        encoder_out2, encoderlayers_out, enc_self_attns = self.encoder(enc_inputs)# encoder_out: [batch_size, 196, 2048]

        encoder_out = encoderlayers_out[0]
        # FIXME:famhshi1
        # encoderout = self.w1(encoderlayers_out[0])+self.w2(encoderlayers_out[1])+self.w3(encoderlayers_out[2])
        # FIXME:fangshi2
        # for i in range(len(encoderlayers_out)-1):
        #     encoder_out = self.encoder_aggregation[i](encoderlayers_out[i+1],encoder_out)
        # FIXME:fangshi5
        # for i in range(len(encoderlayers_out) - 1):
        #     encoder_out = self.encoder_aggregation[i](encoderlayers_out[i + 1], encoder_out)
        # FIXME:fangshi4
        # juhe0 = torch.cat([encoderlayers_out[0].unsqueeze(dim=3),encoderlayers_out[1].unsqueeze(dim=3),encoderlayers_out[2].unsqueeze(dim=3)],dim=3)
        # juhe1 = juhe0.permute(0, 3, 1, 2)
        # encoderout = (((self.conv_5(juhe1))).permute(0, 2, 3, 1)).squeeze(dim=3)

        juhe0 = torch.cat([encoderlayers_out[0].view(batch_size, 14,14, encoder_dim),encoderlayers_out[1].view(batch_size, 14,14, encoder_dim),encoderlayers_out[2].view(batch_size, 14,14, encoder_dim)],dim=3)
        juhe1 = juhe0.permute(0, 3,1,2)
        encoderout = (((self.conv_5(juhe1))).permute(0, 2, 3, 1)).squeeze(dim=3)
        encoderout = encoderout.view(batch_size, -1, encoder_dim)


        # encoderout_conv1 = ((self.conv_1(encoderlayers_out[0].view(batch_size, 14,14, encoder_dim).permute(0, 3,1,2))).permute(0, 2, 3, 1))#.view(batch_size, -1, encoder_dim)
        # encoderout_conv2 = ((self.conv_2(encoderlayers_out[1].view(batch_size, 14, 14, encoder_dim).permute(0, 3,1,2))).permute(0, 2, 3, 1))#.view(batch_size, -1, encoder_dim)
        # encoderout_conv3 = ((self.conv_3(encoderlayers_out[2].view(batch_size, 14, 14, encoder_dim).permute(0, 3,1,2))).permute(0, 2, 3, 1))#.view(batch_size, -1, encoder_dim)
        # encoderout = (torch.cat([encoderout_conv1,encoderout_conv2,encoderout_conv3],dim=3)).view(batch_size, -1, encoder_dim)


        # FIXME:rongheLSTM
        # h, c = self.init_hidden_state(enc_inputs)
        # encoderout2 = torch.zeros_like(encoder_out)
        # for k in range(196):
        #     for t in range(len(encoderlayers_out) ):
        #         h, c = self.decode_step(encoderlayers_out[t][:,k,:],(h, c))
        #     preds = self.fc(self.dropout(h))  # (batch_size_t, vocab_size)
        #     encoderout2[:, k, :] = preds

        predictions, encoded_captions, decode_lengths, sort_ind, dec_self_attns, dec_enc_attns = self.decoder(encoderout, encoded_captions, caption_lengths)
        alphas = {"enc_self_attns": enc_self_attns, "dec_self_attns": dec_self_attns, "dec_enc_attns": dec_enc_attns}

        return predictions, encoded_captions, decode_lengths, alphas, sort_ind




