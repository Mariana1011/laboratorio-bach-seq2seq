import torch
import torch.nn as nn


# Configuración común del laboratorio
VOCAB_SIZE = 128
NUM_VOICES = 4
CONTEXT_STEPS = 32
OUTPUT_STEPS = 16
EMBED_DIM = 16
NUM_LAYERS = 1


# =========================================================
# E1 - Encoder-Decoder LSTM
# =========================================================

class EncoderLSTM(nn.Module):
    def __init__(
        self,
        vocab_size=128,
        embed_dim=16,
        hidden_dim=128,
        num_layers=1
    ):
        super().__init__()

        self.embeddings = nn.ModuleList([
            nn.Embedding(vocab_size, embed_dim)
            for _ in range(NUM_VOICES)
        ])

        self.lstm = nn.LSTM(
            input_size=embed_dim * NUM_VOICES,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )

    def forward(self, x):
        emb = [
            self.embeddings[v](x[:, :, v])
            for v in range(NUM_VOICES)
        ]

        x_emb = torch.cat(emb, dim=-1)

        outputs, (h, c) = self.lstm(x_emb)

        return outputs, h, c


class DecoderLSTM(nn.Module):
    def __init__(
        self,
        vocab_size=128,
        embed_dim=16,
        hidden_dim=128,
        num_layers=1
    ):
        super().__init__()

        self.embeddings = nn.ModuleList([
            nn.Embedding(vocab_size, embed_dim)
            for _ in range(NUM_VOICES)
        ])

        self.lstm = nn.LSTM(
            input_size=embed_dim * NUM_VOICES,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )

        self.output_layers = nn.ModuleList([
            nn.Linear(hidden_dim, vocab_size)
            for _ in range(NUM_VOICES)
        ])

    def forward_step(self, x_step, h, c):
        emb = [
            self.embeddings[v](x_step[:, v])
            for v in range(NUM_VOICES)
        ]

        x_emb = torch.cat(emb, dim=-1)
        x_emb = x_emb.unsqueeze(1)

        output, (h, c) = self.lstm(x_emb, (h, c))

        output = output[:, 0, :]

        logits = torch.stack(
            [layer(output) for layer in self.output_layers],
            dim=1
        )

        return logits, h, c


class Seq2SeqLSTM(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = EncoderLSTM()
        self.decoder = DecoderLSTM()

    def forward(self, x, y=None, teacher_forcing=True):
        encoder_outputs, h, c = self.encoder(x)

        decoder_input = x[:, -1, :]

        logits_steps = []

        for t in range(OUTPUT_STEPS):
            logits, h, c = self.decoder.forward_step(
                decoder_input,
                h,
                c
            )

            logits_steps.append(logits)

            if teacher_forcing and y is not None:
                decoder_input = y[:, t, :]
            else:
                decoder_input = logits.argmax(dim=-1)

        return torch.stack(logits_steps, dim=1)


# =========================================================
# E2 - Encoder-Decoder GRU
# =========================================================

class EncoderGRU(nn.Module):
    def __init__(
        self,
        vocab_size=128,
        embed_dim=16,
        hidden_dim=128,
        num_layers=1
    ):
        super().__init__()

        self.embeddings = nn.ModuleList([
            nn.Embedding(vocab_size, embed_dim)
            for _ in range(NUM_VOICES)
        ])

        self.gru = nn.GRU(
            input_size=embed_dim * NUM_VOICES,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )

    def forward(self, x):
        emb = [
            self.embeddings[v](x[:, :, v])
            for v in range(NUM_VOICES)
        ]

        x_emb = torch.cat(emb, dim=-1)

        outputs, h = self.gru(x_emb)

        return outputs, h


class DecoderGRU(nn.Module):
    def __init__(
        self,
        vocab_size=128,
        embed_dim=16,
        hidden_dim=128,
        num_layers=1
    ):
        super().__init__()

        self.embeddings = nn.ModuleList([
            nn.Embedding(vocab_size, embed_dim)
            for _ in range(NUM_VOICES)
        ])

        self.gru = nn.GRU(
            input_size=embed_dim * NUM_VOICES,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )

        self.output_layers = nn.ModuleList([
            nn.Linear(hidden_dim, vocab_size)
            for _ in range(NUM_VOICES)
        ])

    def forward_step(self, x_step, h):
        emb = [
            self.embeddings[v](x_step[:, v])
            for v in range(NUM_VOICES)
        ]

        x_emb = torch.cat(emb, dim=-1)
        x_emb = x_emb.unsqueeze(1)

        output, h = self.gru(x_emb, h)

        output = output[:, 0, :]

        logits = torch.stack(
            [layer(output) for layer in self.output_layers],
            dim=1
        )

        return logits, h


class Seq2SeqGRU(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = EncoderGRU()
        self.decoder = DecoderGRU()

    def forward(self, x, y=None, teacher_forcing=True):
        encoder_outputs, h = self.encoder(x)

        decoder_input = x[:, -1, :]

        logits_steps = []

        for t in range(OUTPUT_STEPS):
            logits, h = self.decoder.forward_step(
                decoder_input,
                h
            )

            logits_steps.append(logits)

            if teacher_forcing and y is not None:
                decoder_input = y[:, t, :]
            else:
                decoder_input = logits.argmax(dim=-1)

        return torch.stack(logits_steps, dim=1)


# =========================================================
# E3 - GRU con atención aditiva
# =========================================================

class AdditiveAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()

        self.W_enc = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_dec = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, encoder_outputs, decoder_hidden):
        dec = self.W_dec(decoder_hidden).unsqueeze(1)

        scores = self.v(
            torch.tanh(
                self.W_enc(encoder_outputs) + dec
            )
        ).squeeze(-1)

        attention_weights = torch.softmax(scores, dim=1)

        context = torch.bmm(
            attention_weights.unsqueeze(1),
            encoder_outputs
        ).squeeze(1)

        return context, attention_weights


class DecoderGRUAttention(nn.Module):
    def __init__(
        self,
        vocab_size=128,
        embed_dim=16,
        hidden_dim=128,
        num_layers=1
    ):
        super().__init__()

        self.embeddings = nn.ModuleList([
            nn.Embedding(vocab_size, embed_dim)
            for _ in range(NUM_VOICES)
        ])

        self.attention = AdditiveAttention(hidden_dim)

        self.gru = nn.GRU(
            input_size=(embed_dim * NUM_VOICES) + hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )

        self.output_layers = nn.ModuleList([
            nn.Linear(hidden_dim, vocab_size)
            for _ in range(NUM_VOICES)
        ])

    def forward_step(
        self,
        x_step,
        h,
        encoder_outputs
    ):
        emb = [
            self.embeddings[v](x_step[:, v])
            for v in range(NUM_VOICES)
        ]

        x_emb = torch.cat(emb, dim=-1)

        decoder_hidden = h[-1]

        context, attention_weights = self.attention(
            encoder_outputs,
            decoder_hidden
        )

        gru_input = torch.cat(
            [x_emb, context],
            dim=-1
        ).unsqueeze(1)

        output, h = self.gru(
            gru_input,
            h
        )

        output = output[:, 0, :]

        logits = torch.stack(
            [layer(output) for layer in self.output_layers],
            dim=1
        )

        return logits, h, attention_weights


class Seq2SeqGRUAttention(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = EncoderGRU()
        self.decoder = DecoderGRUAttention()

    def forward(
        self,
        x,
        y=None,
        teacher_forcing=True,
        return_attention=False
    ):
        encoder_outputs, h = self.encoder(x)

        decoder_input = x[:, -1, :]

        logits_steps = []
        attention_steps = []

        for t in range(OUTPUT_STEPS):
            logits, h, attn = self.decoder.forward_step(
                decoder_input,
                h,
                encoder_outputs
            )

            logits_steps.append(logits)
            attention_steps.append(attn)

            if teacher_forcing and y is not None:
                decoder_input = y[:, t, :]
            else:
                decoder_input = logits.argmax(dim=-1)

        logits_all = torch.stack(
            logits_steps,
            dim=1
        )

        if return_attention:
            attention_all = torch.stack(
                attention_steps,
                dim=1
            )

            return logits_all, attention_all

        return logits_all


# =========================================================
# E4 - GRU configurable
# =========================================================

class Seq2SeqGRUConfig(nn.Module):
    def __init__(self, hidden_dim=128):
        super().__init__()

        self.encoder = EncoderGRU(
            vocab_size=VOCAB_SIZE,
            embed_dim=EMBED_DIM,
            hidden_dim=hidden_dim,
            num_layers=NUM_LAYERS
        )

        self.decoder = DecoderGRU(
            vocab_size=VOCAB_SIZE,
            embed_dim=EMBED_DIM,
            hidden_dim=hidden_dim,
            num_layers=NUM_LAYERS
        )

    def forward(self, x, y=None, teacher_forcing=True):
        encoder_outputs, h = self.encoder(x)

        decoder_input = x[:, -1, :]

        logits_steps = []

        for t in range(OUTPUT_STEPS):
            logits, h = self.decoder.forward_step(
                decoder_input,
                h
            )

            logits_steps.append(logits)

            if teacher_forcing and y is not None:
                decoder_input = y[:, t, :]
            else:
                decoder_input = logits.argmax(dim=-1)

        return torch.stack(logits_steps, dim=1)
