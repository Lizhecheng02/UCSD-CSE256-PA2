import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_sequence_length):
        super().__init__()
        self.max_sequence_length = max_sequence_length
        self.d_model = d_model

    def forward(self):
        even_i = torch.arange(0, self.d_model, 2).float()
        denominator = torch.pow(10000, even_i / self.d_model)
        position = (torch.arange(self.max_sequence_length).reshape(self.max_sequence_length, 1))
        even_positional_embedding = torch.sin(position / denominator)
        odd_positional_embedding = torch.cos(position / denominator)
        stacked = torch.stack([even_positional_embedding, odd_positional_embedding], dim=2)
        positional_embedding = torch.flatten(stacked, start_dim=1, end_dim=2)
        return positional_embedding


def alibi_encoding(seq_len, num_heads):
    distance_matrix = torch.arange(seq_len).unsqueeze(0) - torch.arange(seq_len).unsqueeze(1)
    alibi = distance_matrix.unsqueeze(0).repeat(num_heads, 1, 1)
    return alibi


class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        assert (self.head_dim * heads == embed_size), "Embedding size needs to be divisible by heads"

        self.values = nn.Linear(embed_size, embed_size)
        self.keys = nn.Linear(embed_size, embed_size)
        self.queries = nn.Linear(embed_size, embed_size)
        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, value, key, query, mask, alibi_bias=None):
        N = query.shape[0]

        value_len, key_len, query_len = value.shape[1], key.shape[1], query.shape[1]
        values = self.values(value)
        keys = self.keys(key)
        queries = self.queries(query)

        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = queries.reshape(N, query_len, self.heads, self.head_dim)

        qk = torch.einsum("nqhd, nkhd -> nhqk", [queries, keys])

        if mask is not None:
            qk = qk.masked_fill(mask == 0, float("-inf"))

        attention = qk / (self.embed_size ** (1 / 2))
        if alibi_bias is not None:
            attention = attention + alibi_bias
        attention = torch.softmax(attention, dim=3)

        out = torch.einsum("nhqk, nkhd -> nqhd", [attention, values]).reshape(N, query_len, self.heads * self.head_dim)
        out = self.fc_out(out)

        return out, attention


class DeBERTaDisentangledAttention(nn.Module):
    def __init__(self, embed_size, heads, max_position_embeddings):
        super().__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        assert (self.head_dim * heads == embed_size), "Embedding size needs to be divisible by heads"
        
        self.query = nn.Linear(embed_size, embed_size)
        self.key_content = nn.Linear(embed_size, embed_size)
        self.key_position = nn.Linear(embed_size, embed_size)
        self.value = nn.Linear(embed_size, embed_size)
        self.output = nn.Linear(embed_size, embed_size)
        self.fc_out = nn.Linear(embed_size, embed_size)

        self.position_embedding = nn.Embedding(2 * max_position_embeddings, embed_size)
        self.max_position_embeddings = max_position_embeddings

    def forward(self, value, key, query, mask):
        N = query.shape[0]
        value_len, key_len, query_len = value.shape[1], key.shape[1], query.shape[1]
    
        position_ids = torch.arange(query_len, dtype=torch.long, device=query.device)
        relative_position_matrix = position_ids[None, :] - position_ids[:, None]
        relative_position_matrix += self.max_position_embeddings
        position_embeddings = self.position_embedding(relative_position_matrix)
        
        values = self.value(value).reshape(N, value_len, self.heads, self.head_dim)
        queries = self.query(query).reshape(N, query_len, self.heads, self.head_dim)
        keys_content = self.key_content(key).reshape(N, key_len, self.heads, self.head_dim)
        keys_position = self.key_position(position_embeddings).reshape(1, query_len, query_len, self.heads, self.head_dim)
        
        attention_scores_content = torch.einsum("nqhd, nkhd -> nhqk", [queries, keys_content])
        attention_scores_position = torch.einsum("nqhd, nqkhd -> nhqk", [queries, keys_position])

        attention_scores = attention_scores_content + attention_scores_position
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, float("-inf"))

        attention = torch.softmax(attention_scores / (self.embed_size ** 0.5), dim=-1)

        out = torch.einsum("nhqk, nkhd -> nqhd", [attention, values]).reshape(N, query_len, self.heads * self.head_dim)
        out = self.fc_out(out)

        return out, attention
    

class WindowedSelfAttention(nn.Module):
    def __init__(self, embed_size, heads, window_size):
        super(WindowedSelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        self.window_size = window_size
        assert (self.head_dim * heads == embed_size), "Embedding size needs to be divisible by heads"

        self.values = nn.Linear(embed_size, embed_size)
        self.keys = nn.Linear(embed_size, embed_size)
        self.queries = nn.Linear(embed_size, embed_size)
        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, value, key, query, mask, alibi_bias=None):
        N = query.shape[0]
        value_len, key_len, query_len = value.shape[1], key.shape[1], query.shape[1]
        window_size = min(self.window_size, key_len)

        values = self.values(value)
        keys = self.keys(key)
        queries = self.queries(query)

        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = queries.reshape(N, query_len, self.heads, self.head_dim)

        attention = torch.zeros(N, self.heads, query_len, query_len).to(queries.device)

        for i in range(query_len):
            start_index = max(0, i - window_size // 2)
            end_index = min(key_len, i + window_size // 2 + 1)
            qk_window = torch.einsum("nqhd, nkhd -> nhqk", [queries[:, i:i + 1], keys[:, start_index:end_index]])
            if mask is not None:
                mask_window = mask[:, :, :, start_index:end_index]
                qk_window = qk_window.masked_fill(mask_window == 0, float("-inf"))

            qk_window = qk_window.squeeze(2)
            attention[:, :, i, start_index:end_index] = qk_window

        attention = attention / (self.embed_size ** (1 / 2))
        # print(attention.shape)
        if alibi_bias is not None:
            attention += alibi_bias

        attention = torch.softmax(attention, dim=3)
        out = torch.einsum("nhqk, nkhd -> nqhd", [attention, values]).reshape(N, query_len, self.heads * self.head_dim)
        out = self.fc_out(out)

        return out, attention
    

class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, feed_forward_dimension):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, feed_forward_dimension),
            nn.ReLU(),
            nn.Linear(feed_forward_dimension, embed_size)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask, alibi_bias=None):
        output, attention = self.attention(value, key, query, mask, alibi_bias)
        x = self.dropout(self.norm1(output + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out, attention
    

class TransformerBlockDeberta(nn.Module):
    def __init__(self, embed_size, heads, dropout, feed_forward_dimension):
        super(TransformerBlockDeberta, self).__init__()
        self.attention = DeBERTaDisentangledAttention(embed_size, heads, max_position_embeddings=32)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, feed_forward_dimension),
            nn.ReLU(),
            nn.Linear(feed_forward_dimension, embed_size)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):
        output, attention = self.attention(value, key, query, mask)
        x = self.dropout(self.norm1(output + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out, attention


class TransformerBlockWindowAttention(nn.Module):
    def __init__(self, embed_size, heads, dropout, feed_forward_dimension, window_size):
        super(TransformerBlockWindowAttention, self).__init__()
        self.attention = WindowedSelfAttention(embed_size, heads, window_size)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, feed_forward_dimension),
            nn.ReLU(),
            nn.Linear(feed_forward_dimension, embed_size)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask, alibi_bias=None):
        output, attention = self.attention(value, key, query, mask, alibi_bias)
        x = self.dropout(self.norm1(output + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out, attention


class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_size, num_layers, heads, device, feed_forward_dimension, dropout, max_length):
        super(Decoder, self).__init__()
        self.embed_size = embed_size
        self.word_embedding = nn.Embedding(vocab_size, embed_size)
        self.position_embedding = PositionalEncoding(embed_size, max_length)
        self.layers = nn.ModuleList([TransformerBlock(embed_size, heads, dropout, feed_forward_dimension) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(embed_size)
        self.fc_out = nn.Linear(embed_size, vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.device = device

    def make_mask(self, input_ids):
        N, seq_length = input_ids.shape
        mask = torch.tril(torch.ones((seq_length, seq_length))).expand(N, 1, seq_length, seq_length)
        # print(mask)
        return mask.to(self.device)

    def forward(self, x):
        N, seq_length = x.shape
        mask = self.make_mask(x)
        pos_embed = self.position_embedding().unsqueeze(0).expand(N, -1, -1)[:, :seq_length, :]
        out = self.dropout(self.word_embedding(x) + pos_embed)
        attention_matrices = []
        for layer in self.layers:
            out, attention = layer(out, out, out, mask)
            attention_matrices.append(attention.detach().cpu())
        out = self.fc_out(self.norm(out))
        return out, attention_matrices


class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_size, num_layers, heads, device, feed_forward_dimension, dropout, max_length, pad_idx):
        super(Encoder, self).__init__()
        self.embed_size = embed_size
        self.device = device
        self.word_embedding = nn.Embedding(vocab_size, embed_size)
        # self.position_embedding = nn.Embedding(max_length, embed_size)
        self.position_embedding = PositionalEncoding(embed_size, max_length)
        self.layers = nn.ModuleList([TransformerBlock(embed_size, heads, dropout, feed_forward_dimension) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
        self.pad_idx = pad_idx

    def make_mask(self, input_ids):
        src_mask = (input_ids != self.pad_idx).unsqueeze(1).unsqueeze(2)
        print(src_mask)
        return src_mask.to(self.device)

    def forward(self, x):
        N, seq_length = x.shape
        mask = self.make_mask(x)
        # positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        pos_embed = self.position_embedding().unsqueeze(0).expand(N, -1, -1)[:, :seq_length, :]
        # out = self.dropout((self.word_embedding(x) + self.position_embedding(positions)))
        attention_matrices = []
        out = self.dropout((self.word_embedding(x) + pos_embed))
        for layer in self.layers:
            out, attention = layer(out, out, out, mask)
            attention_matrices.append(attention.detach().cpu())
        # print(attention_matrices.shape)
        return out, attention_matrices


class ClassificationEncoder(nn.Module):
    def __init__(self, vocab_size, embed_size, num_layers, heads, device, feed_forward_dimension, dropout, max_length, pad_idx, cls_hidden_size, num_classes):
        super(ClassificationEncoder, self).__init__()
        self.embed_size = embed_size
        self.device = device
        self.word_embedding = nn.Embedding(vocab_size, embed_size)
        # self.position_embedding = nn.Embedding(max_length, embed_size)
        self.position_embedding = PositionalEncoding(embed_size, max_length)
        self.layers = nn.ModuleList([TransformerBlock(embed_size, heads, dropout, feed_forward_dimension) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
        self.classification_hidden = nn.Linear(embed_size, cls_hidden_size)
        self.classification_head = nn.Linear(cls_hidden_size, num_classes)
        self.pad_idx = pad_idx

    def make_src_mask(self, src):
        src_mask = (src != self.pad_idx).unsqueeze(1).unsqueeze(2)
        # print(src_mask)
        return src_mask.to(self.device)

    def forward(self, x):
        N, seq_length = x.shape
        mask = self.make_src_mask(x)
        # print(mask)
        pos_embed = self.position_embedding().unsqueeze(0).expand(N, -1, -1)[:, :seq_length, :]
        attention_matrices = []
        out = self.dropout((self.word_embedding(x) + pos_embed))
        for layer in self.layers:
            out, attention = layer(out, out, out, mask)
            attention_matrices.append(attention.detach().cpu())
        out = out.mean(dim=1)
        out = self.classification_hidden(out)
        out = self.classification_head(out)
        return out, attention_matrices
    

class ClassificationEncoderCLSToken(nn.Module):
    def __init__(self, vocab_size, embed_size, num_layers, heads, device, feed_forward_dimension, dropout, max_length, pad_idx, cls_hidden_size, num_classes):
        super(ClassificationEncoderCLSToken, self).__init__()
        self.embed_size = embed_size
        self.device = device
        self.word_embedding = nn.Embedding(vocab_size, embed_size)
        # self.position_embedding = nn.Embedding(max_length, embed_size)
        self.position_embedding = PositionalEncoding(embed_size, max_length)
        self.layers = nn.ModuleList([TransformerBlock(embed_size, heads, dropout, feed_forward_dimension) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
        self.classification_hidden = nn.Linear(embed_size, cls_hidden_size)
        self.classification_head = nn.Linear(cls_hidden_size, num_classes)
        self.pad_idx = pad_idx

    def make_src_mask(self, src):
        src_mask = (src != self.pad_idx).unsqueeze(1).unsqueeze(2)
        # print(src_mask)
        return src_mask.to(self.device)

    def forward(self, x):
        N, seq_length = x.shape
        mask = self.make_src_mask(x)
        # print(mask)
        pos_embed = self.position_embedding().unsqueeze(0).expand(N, -1, -1)[:, :seq_length, :]
        attention_matrices = []
        out = self.dropout((self.word_embedding(x) + pos_embed))
        for layer in self.layers:
            out, attention = layer(out, out, out, mask)
            attention_matrices.append(attention.detach().cpu())
        cls_embedding = out[:, 0, :]
        out = self.classification_hidden(cls_embedding)
        out = self.classification_head(out)
        return out, attention_matrices


class ClassificationEncoderDeberta(nn.Module):
    def __init__(self, vocab_size, embed_size, num_layers, heads, device, feed_forward_dimension, dropout, max_length, pad_idx, cls_hidden_size, num_classes):
        super(ClassificationEncoderDeberta, self).__init__()
        self.embed_size = embed_size
        self.device = device
        self.word_embedding = nn.Embedding(vocab_size, embed_size)
        # self.position_embedding = nn.Embedding(max_length, embed_size)
        self.position_embedding = PositionalEncoding(embed_size, max_length)
        self.layers = nn.ModuleList([TransformerBlockDeberta(embed_size, heads, dropout, feed_forward_dimension) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
        self.classification_hidden = nn.Linear(embed_size, cls_hidden_size)
        self.classification_head = nn.Linear(cls_hidden_size, num_classes)
        self.pad_idx = pad_idx

    def make_src_mask(self, src):
        src_mask = (src != self.pad_idx).unsqueeze(1).unsqueeze(2)
        # print(src_mask)
        return src_mask.to(self.device)

    def forward(self, x):
        N, seq_length = x.shape
        mask = self.make_src_mask(x)
        # print(mask)
        pos_embed = self.position_embedding().unsqueeze(0).expand(N, -1, -1)[:, :seq_length, :]
        attention_matrices = []
        out = self.dropout((self.word_embedding(x) + pos_embed))
        for layer in self.layers:
            out, attention = layer(out, out, out, mask)
            attention_matrices.append(attention.detach().cpu())
        out = out.mean(dim=1)
        out = self.classification_hidden(out)
        out = self.classification_head(out)
        return out, attention_matrices


class ClassificationEncoderWindowAttention(nn.Module):
    def __init__(self, vocab_size, embed_size, num_layers, heads, device, feed_forward_dimension, dropout, max_length, pad_idx, cls_hidden_size, num_classes, window_size=16):
        super(ClassificationEncoderWindowAttention, self).__init__()
        self.embed_size = embed_size
        self.device = device
        self.word_embedding = nn.Embedding(vocab_size, embed_size)
        # self.position_embedding = nn.Embedding(max_length, embed_size)
        self.position_embedding = PositionalEncoding(embed_size, max_length)
        self.layers = nn.ModuleList([TransformerBlockWindowAttention(embed_size, heads, dropout, feed_forward_dimension, window_size) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
        self.classification_hidden = nn.Linear(embed_size, cls_hidden_size)
        self.classification_head = nn.Linear(cls_hidden_size, num_classes)
        self.pad_idx = pad_idx

    def make_src_mask(self, src):
        src_mask = (src != self.pad_idx).unsqueeze(1).unsqueeze(2)
        # print(src_mask)
        return src_mask.to(self.device)

    def forward(self, x):
        N, seq_length = x.shape
        mask = self.make_src_mask(x)
        pos_embed = self.position_embedding().unsqueeze(0).expand(N, -1, -1)[:, :seq_length, :]
        attention_matrices = []
        out = self.dropout((self.word_embedding(x) + pos_embed))
        for layer in self.layers:
            out, attention = layer(out, out, out, mask)
            attention_matrices.append(attention.detach().cpu())
        out = out.mean(dim=1)
        out = self.classification_hidden(out)
        out = self.classification_head(out)
        return out, attention_matrices


class ClassificationEncoderAlibi(nn.Module):
    def __init__(self, vocab_size, embed_size, num_layers, heads, device, feed_forward_dimension, dropout, max_length, pad_idx, cls_hidden_size, num_classes):
        super(ClassificationEncoderAlibi, self).__init__()
        self.embed_size = embed_size
        self.device = device
        self.word_embedding = nn.Embedding(vocab_size, embed_size)
        self.layers = nn.ModuleList([TransformerBlock(embed_size, heads, dropout, feed_forward_dimension) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
        self.classification_hidden = nn.Linear(embed_size, cls_hidden_size)
        self.classification_head = nn.Linear(cls_hidden_size, num_classes)
        self.pad_idx = pad_idx
        self.heads = heads

    def make_src_mask(self, src):
        src_mask = (src != self.pad_idx).unsqueeze(1).unsqueeze(2)
        # print(src_mask)
        return src_mask.to(self.device)

    def forward(self, x):
        N, seq_length = x.shape
        alibi_bias = alibi_encoding(seq_len=seq_length, num_heads=self.heads)
        mask = self.make_src_mask(x)
        attention_matrices = []
        out = self.dropout(self.word_embedding(x))
        for layer in self.layers:
            out, attention = layer(out, out, out, mask, alibi_bias=alibi_bias)
            attention_matrices.append(attention.detach().cpu())
        out = out.mean(dim=1)
        out = self.classification_hidden(out)
        out = self.classification_head(out)
        return out, attention_matrices


if __name__ == "__main__":
    encoder_model = Encoder(
        vocab_size=1000,
        embed_size=64,
        num_layers=2,
        heads=2,
        device="cpu",
        feed_forward_dimension=100,
        dropout=0.1,
        max_length=16,
        pad_idx=0
    )
    output, attention_matrices = encoder_model(torch.tensor([[1, 2, 3, 4, 5, 0, 0, 0], [6, 7, 8, 9, 10, 0, 0, 0]]))
    print(output.shape)
    print(len(attention_matrices))

    encoder_model = ClassificationEncoder(
        vocab_size=1000,
        embed_size=64,
        num_layers=2,
        heads=2,
        device="cpu",
        feed_forward_dimension=100,
        dropout=0.1,
        max_length=16,
        pad_idx=0,
        cls_hidden_size=50,
        num_classes=3
    )
    output = encoder_model(torch.tensor([[1, 2, 3, 4, 5, 0, 0, 0], [6, 7, 8, 9, 10, 0, 0, 0]]))
    print(output)
    print(output.shape)
    print(torch.max(output.data, 1))

    decoder_model = Decoder(
        vocab_size=1000,
        embed_size=64,
        num_layers=2,
        heads=2,
        device="cpu",
        feed_forward_dimension=4,
        dropout=0.1,
        max_length=16
    )
    output = decoder_model(torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]))
    print(output)
    print(output.shape)