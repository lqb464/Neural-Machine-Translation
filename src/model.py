import torch
import torch.nn as nn
import torch.nn.functional as F
import random


class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_layers=1):
        super(Encoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.gru = nn.GRU(embed_dim, hidden_dim, n_layers, batch_first=True)

    def forward(self, x):
        embedded = self.embedding(x)
        outputs, hidden = self.gru(embedded)
        return outputs, hidden


class LuongAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(LuongAttention, self).__init__()
        self.wa = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, decoder_hidden, encoder_outputs):
        decoder_hidden_permuted = decoder_hidden.permute(1, 2, 0)
        score = self.wa(encoder_outputs)
        scores = torch.bmm(score, decoder_hidden_permuted)
        attention_weights = F.softmax(scores, dim=1)
        context = torch.bmm(attention_weights.permute(0, 2, 1), encoder_outputs)
        return context, attention_weights


class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_layers=1):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.gru = nn.GRU(embed_dim + hidden_dim, hidden_dim, n_layers, batch_first=True)
        self.attention = LuongAttention(hidden_dim)
        self.out = nn.Linear(hidden_dim * 2, vocab_size)

    def forward(self, x, hidden, encoder_outputs):
        embedded = self.embedding(x)
        context, attn_weights = self.attention(hidden, encoder_outputs)
        rnn_input = torch.cat((embedded, context), dim=2)
        output, hidden = self.gru(rnn_input, hidden)
        output = torch.cat((output, context), dim=2)
        prediction = self.out(output.squeeze(1))
        return prediction, hidden, attn_weights


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, tgt, teacher_forcing_ratio=0.5):
        batch_size = src.shape[0]
        max_len = tgt.shape[1]
        tgt_vocab_size = self.decoder.out.out_features

        outputs = torch.zeros(batch_size, max_len, tgt_vocab_size, device=self.device)

        encoder_outputs, hidden = self.encoder(src)
        decoder_input = tgt[:, 0].unsqueeze(1)

        for t in range(1, max_len):
            prediction, hidden, _ = self.decoder(decoder_input, hidden, encoder_outputs)
            outputs[:, t, :] = prediction

            top1 = prediction.argmax(1)
            use_teacher_forcing = random.random() < teacher_forcing_ratio
            decoder_input = tgt[:, t].unsqueeze(1) if use_teacher_forcing else top1.unsqueeze(1)

        return outputs