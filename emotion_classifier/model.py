import torch.nn as nn

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim * 2, 1)  # for bidirectional

    def forward(self, lstm_out):
        weights = self.attn(lstm_out).squeeze(-1)         # [batch, seq_len]
        weights = torch.softmax(weights, dim=1)           # attention weights
        context = torch.sum(lstm_out * weights.unsqueeze(-1), dim=1)  # [batch, hidden*2]
        return context

class MyModel(nn.Module):
    def __init__(self, embedding_matrix, hidden_dim, output_dim, pad_idx, dropout=0.3):
        super(MyModel, self).__init__()
        vocab_size, embed_dim = embedding_matrix.shape
        self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=False, padding_idx=pad_idx)
        self.rnn = nn.GRU(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.attention = Attention(hidden_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.projection = nn.Linear(hidden_dim * 2, 64)
        self.fc = nn.Linear(64, output_dim)

    def forward(self, x):
        emb = self.embedding(x)
        rnn_out, _ = self.rnn(emb)
        context = self.attention(rnn_out)
        context = self.dropout(context)
        context = self.projection(context)
        return self.fc(context)