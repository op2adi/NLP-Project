# baseline2.py — BiLSTM+MLP for MultiBooked Sentiment Graph Parsing with GPU support

import json
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score
from tqdm import tqdm
import numpy as np
import random
import os

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------- Data Utilities -----------
def load_data(path):
    with open(path, 'r') as f:
        return json.load(f)

def tokenize(text):
    return text.strip().split()

def build_vocab(data):
    vocab = {'<pad>': 0, '<unk>': 1}
    for entry in data:
        for word in tokenize(entry['text']):
            if word not in vocab:
                vocab[word] = len(vocab)
    return vocab

def encode(sentence, vocab, max_len=50):
    tokens = tokenize(sentence)
    ids = [vocab.get(t, vocab['<unk>']) for t in tokens][:max_len]
    return ids + [vocab['<pad>']] * (max_len - len(ids))

def span_to_token_index(span_str, tokens, text):
    if not span_str:
        return -1
    try:
        start, end = map(int, span_str.split(':'))
        char_pos = 0
        for i, token in enumerate(tokens):
            while char_pos < len(text) and text[char_pos].isspace():
                char_pos += 1
            token_start = char_pos
            token_end = char_pos + len(token)
            if token_start <= start < token_end:
                return i
            char_pos = token_end
        return -1
    except:
        return -1

def create_examples(data, vocab, max_len=50):
    examples = []
    for entry in data:
        text = entry['text']
        tokens = tokenize(text)
        sent_ids = encode(text, vocab, max_len)
        edges = set()

        for op in entry.get('opinions', []):
            t_str = op['Target'][1][0] if op['Target'][1] else ''
            p_str = op['Polar_expression'][1][0] if op['Polar_expression'][1] else ''
            label = op.get('Polarity', 'Neutral')

            from_idx = span_to_token_index(p_str, tokens, text)
            to_idx = span_to_token_index(t_str, tokens, text)

            if 0 <= from_idx < max_len and 0 <= to_idx < max_len:
                edges.add((from_idx, to_idx, label))

        examples.append((sent_ids, edges))
    return examples

# ----------- Model -----------
class BiLSTMBaseline(nn.Module):
    def __init__(self, vocab_size, embed_dim=100, hidden_dim=64, num_labels=3):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.edge_mlp = nn.Linear(hidden_dim * 4, 1)
        self.label_mlp = nn.Linear(hidden_dim * 4, num_labels)

    def forward(self, x):
        emb = self.embed(x)
        out, _ = self.lstm(emb)
        B, T, D = out.size()
        pair_reps = []
        for i in range(T):
            for j in range(T):
                pair = torch.cat([out[:, i], out[:, j]], dim=-1)
                pair_reps.append(pair)
        pair_reps = torch.stack(pair_reps, dim=1)
        edge_logits = self.edge_mlp(pair_reps).squeeze(-1)
        label_logits = self.label_mlp(pair_reps)
        return edge_logits, label_logits

# ----------- Training & Evaluation -----------
def train_model(model, data, vocab, epochs=5, max_len=50):
    model.to(device)
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()
    label_loss_fn = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        random.shuffle(data)
        total_loss = 0
        for sent, gold_edges in tqdm(data):
            x = torch.tensor([sent]).to(device)
            edge_logits, label_logits = model(x)

            T = len(sent)
            gold_edges_binary = torch.zeros((1, T*T)).to(device)
            gold_labels = torch.zeros((1, T*T), dtype=torch.long).to(device)

            for i in range(T):
                for j in range(T):
                    idx = i * T + j
                    for from_, to_, label in gold_edges:
                        if from_ == i and to_ == j:
                            gold_edges_binary[0, idx] = 1
                            gold_labels[0, idx] = label_to_id(label)

            loss = loss_fn(edge_logits, gold_edges_binary) + label_loss_fn(
                label_logits.view(-1, label_logits.size(-1)), gold_labels.view(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1} Loss: {total_loss / len(data):.4f}")

def evaluate(model, data, vocab, max_len=50):
    model.to(device)
    model.eval()
    y_true, y_pred = [], []
    for sent, gold_edges in data:
        x = torch.tensor([sent]).to(device)
        with torch.no_grad():
            edge_logits, label_logits = model(x)

        T = len(sent)
        edge_preds = (torch.sigmoid(edge_logits) > 0.5).view(T, T)
        label_preds = label_logits.argmax(-1).view(T, T)

        pred_edges = set()
        for i in range(T):
            for j in range(T):
                if edge_preds[i, j]:
                    pred_edges.add((i, j, id_to_label(label_preds[i, j].item())))

        true_edges = set(gold_edges)
        y_true.extend(true_edges)
        y_pred.extend(pred_edges)

    f1 = sentiment_graph_f1(y_true, y_pred)
    print(f"Sentiment Graph F1: {f1:.4f}")

# ----------- Label Encoding -----------
label2id = {'Positive': 0, 'Negative': 1, 'Neutral': 2}
id2label = {v: k for k, v in label2id.items()}

def label_to_id(label):
    return label2id.get(label, 2)

def id_to_label(id):
    return id2label.get(id, 'Neutral')

# ----------- F1 Calculation -----------
def sentiment_graph_f1(y_true, y_pred):
    y_true = set(y_true)
    y_pred = set(y_pred)
    tp = len(y_true & y_pred)
    precision = tp / len(y_pred) if y_pred else 0
    recall = tp / len(y_true) if y_true else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return f1

# ----------- Run Pipeline -----------
if __name__ == '__main__':
    train_path = "opener_en/train.json"
    test_path = "multibooked_ca/dev.json"

    train_raw = load_data(train_path)
    test_raw = load_data(test_path)

    vocab = build_vocab(train_raw)
    train_data = create_examples(train_raw, vocab)
    test_data = create_examples(test_raw, vocab)

    dataset_name = os.path.basename(os.path.dirname(train_path))
    model_path = f"bilstm_mlp_model_{dataset_name}.pt"

    model = BiLSTMBaseline(vocab_size=len(vocab))

    if os.path.exists(model_path):
        print(f"Model already exists at {model_path}. Skipping training.")
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        train_model(model, train_data, vocab, epochs=5)
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")

    evaluate(model, test_data, vocab)
