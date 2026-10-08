import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from .config import (
    OUTPUT_DIR,
    MODEL_DIR,
    MAX_SEQUENCE_LENGTH,
    EMBEDDING_DIM,
    MAX_USERS,
    RANDOM_SEED,
)

torch.manual_seed(RANDOM_SEED)


# ============================================================
# VOCABULARY
# ============================================================

class EventVocabulary:

    def __init__(self):
        self.token2id = {
            "<PAD>": 0,
            "<UNK>": 1
        }

        self.id2token = [
            "<PAD>",
            "<UNK>"
        ]

    def add_token(self, token):

        if token not in self.token2id:
            self.token2id[token] = len(self.id2token)
            self.id2token.append(token)

        return self.token2id[token]

    def __len__(self):
        return len(self.id2token)

    def encode(self, token):

        return self.token2id.get(
            token,
            self.token2id["<UNK>"]
        )

    def decode(self, idx):

        if 0 <= idx < len(self.id2token):
            return self.id2token[idx]

        return "<UNK>"


# ============================================================
# DATASET
# ============================================================

class UserSequenceDataset(Dataset):

    def __init__(
        self,
        sequences_file,
        vocab,
        max_len,
        max_users=MAX_USERS
    ):

        df = pd.read_csv(sequences_file)

        # Limit users for development
        if max_users is not None and len(df) > max_users:

            df = df.sample(
                n=max_users,
                random_state=RANDOM_SEED
            ).reset_index(drop=True)

        self.user_ids = df["visitorid"].tolist()
        self.sequences = df["event_sequence"].fillna("").tolist()

        # ----------------------------------------------------
        # Build vocabulary
        # ----------------------------------------------------

        for seq in self.sequences:

            for token in seq.split():

                vocab.add_token(token)

        self.vocab = vocab
        self.max_len = max_len

        self.encoded = []

        for seq in self.sequences:

            tokens = seq.split()

            ids = [
                vocab.encode(token)
                for token in tokens
            ]

            # Keep most recent events
            if len(ids) > max_len:

                ids = ids[-max_len:]

            # Padding
            if len(ids) < max_len:

                ids += [
                    vocab.encode("<PAD>")
                ] * (max_len - len(ids))

            self.encoded.append(
                np.array(ids, dtype=np.int64)
            )

    def __len__(self):

        return len(self.encoded)

    def __getitem__(self, idx):

        ids = self.encoded[idx]

        return {
            "input_ids": torch.tensor(
                ids,
                dtype=torch.long
            ),
            "user_id": self.user_ids[idx]
        }


# ============================================================
# TRANSFORMER
# ============================================================

class TransformerEncoderModel(nn.Module):

    def __init__(
        self,
        vocab_size,
        embed_dim=EMBEDDING_DIM,
        nhead=2,
        num_layers=1,
        dropout=0.1
    ):

        super().__init__()

        self.embed_dim = embed_dim

        # Event/token embeddings
        self.token_embed = nn.Embedding(
            vocab_size,
            embed_dim,
            padding_idx=0
        )

        # Positional embeddings
        self.pos_embed = nn.Parameter(
            torch.zeros(
                1,
                MAX_SEQUENCE_LENGTH,
                embed_dim
            )
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dropout=dropout,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # VERY SMALL prediction head.
        # We only predict:
        # 0 = view
        # 1 = addtocart
        # 2 = transaction
        self.event_classifier = nn.Linear(
            embed_dim,
            3
        )

    def forward(self, input_ids):

        x = self.token_embed(input_ids)

        x = x + self.pos_embed[
            :,
            :input_ids.size(1),
            :
        ]

        padding_mask = (
            input_ids == 0
        )

        x = self.transformer(
            x,
            src_key_padding_mask=padding_mask
        )

        # Mean pooling over non-padding tokens

        mask = (
            ~padding_mask
        ).unsqueeze(-1)

        masked_x = x * mask

        pooled = masked_x.sum(dim=1) / (
            mask.sum(dim=1).clamp(min=1)
        )

        logits = self.event_classifier(
            pooled
        )

        return logits

    def get_user_embedding(self, input_ids):

        with torch.no_grad():

            x = self.token_embed(
                input_ids
            )

            x = x + self.pos_embed[
                :,
                :input_ids.size(1),
                :
            ]

            padding_mask = (
                input_ids == 0
            )

            x = self.transformer(
                x,
                src_key_padding_mask=padding_mask
            )

            mask = (
                ~padding_mask
            ).unsqueeze(-1)

            masked_x = x * mask

            pooled = masked_x.sum(dim=1) / (
                mask.sum(dim=1).clamp(min=1)
            )

        return pooled


# ============================================================
# TRAINING
# ============================================================

def train_transformer(
    sequences_path,
    epochs=2,
    batch_size=64,
    lr=1e-3
):

    print("\n--- Training lightweight Transformer ---")

    vocab = EventVocabulary()

    dataset = UserSequenceDataset(
        sequences_path,
        vocab,
        MAX_SEQUENCE_LENGTH,
        MAX_USERS
    )

    print(
        f"Training users: {len(dataset)}"
    )

    print(
        f"Vocabulary size: {len(vocab)}"
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True
    )

    model = TransformerEncoderModel(
        vocab_size=len(vocab),
        embed_dim=EMBEDDING_DIM
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr
    )

    criterion = nn.CrossEntropyLoss()

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    for epoch in range(1, epochs + 1):

        model.train()

        total_loss = 0

        for batch in dataloader:

            input_ids = batch["input_ids"]

            # ------------------------------------------------
            # Derive behavioral target from the last event
            # ------------------------------------------------

            targets = []

            for row in input_ids:

                valid_tokens = row[
                    row != 0
                ]

                if len(valid_tokens) == 0:

                    targets.append(0)

                    continue

                token_id = valid_tokens[-1].item()

                token = vocab.decode(
                    token_id
                )

                if token.startswith("view"):

                    targets.append(0)

                elif token.startswith("addtocart"):

                    targets.append(1)

                elif token.startswith("transaction"):

                    targets.append(2)

                else:

                    targets.append(0)

            targets = torch.tensor(
                targets,
                dtype=torch.long
            )

            optimizer.zero_grad()

            logits = model(
                input_ids
            )

            loss = criterion(
                logits,
                targets
            )

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        avg_loss = (
            total_loss /
            len(dataloader)
        )

        print(
            f"Epoch {epoch}/{epochs} "
            f"- Loss: {avg_loss:.4f}"
        )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    checkpoint = {

        "model_state_dict":
            model.state_dict(),

        "vocab":
            vocab,

        "embed_dim":
            EMBEDDING_DIM,

        "max_sequence_length":
            MAX_SEQUENCE_LENGTH
    }

    torch.save(
        checkpoint,
        MODEL_DIR /
        "transformer_embeddings.pt"
    )

    print(
        "\nTransformer model saved."
    )

    return model, vocab


# ============================================================
# LOAD MODEL
# ============================================================

def load_trained_model():

    checkpoint_path = (
        MODEL_DIR /
        "transformer_embeddings.pt"
    )

    if not checkpoint_path.is_file():

        raise FileNotFoundError(
            f"Checkpoint not found at "
            f"{checkpoint_path}"
        )

    data = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False
    )

    vocab = data["vocab"]

    model = TransformerEncoderModel(
        vocab_size=len(vocab),
        embed_dim=EMBEDDING_DIM
    )

    model.load_state_dict(
        data["model_state_dict"]
    )

    model.eval()

    return model, vocab


# ============================================================
# GENERATE USER EMBEDDINGS
# ============================================================

def generate_user_embeddings(
    model,
    vocab,
    sequences_path
):

    print(
        "\n--- Generating Transformer embeddings ---"
    )

    dataset = UserSequenceDataset(
        sequences_path,
        vocab,
        MAX_SEQUENCE_LENGTH,
        MAX_USERS
    )

    loader = DataLoader(
        dataset,
        batch_size=256,
        shuffle=False
    )

    embeddings = []
    user_ids = []

    model.eval()

    with torch.no_grad():

        for batch in loader:

            input_ids = batch[
                "input_ids"
            ]

            emb = model.get_user_embedding(
                input_ids
            )

            embeddings.append(
                emb.cpu().numpy()
            )

            user_ids.extend(
                batch["user_id"]
            )

    embeddings = np.vstack(
        embeddings
    )

    df_emb = pd.DataFrame(
        embeddings,
        columns=[
            f"emb_{i}"
            for i in range(EMBEDDING_DIM)
        ]
    )

    df_emb["visitorid"] = user_ids

    output_path = (
        OUTPUT_DIR /
        "user_embeddings.csv"
    )

    df_emb.to_csv(
        output_path,
        index=False
    )

    print(
        f"Saved embeddings to {output_path}"
    )

    return df_emb


# ============================================================
# VISUALIZATION
# ============================================================

def visualize_embeddings(
    df_emb,
    n_components=2,
    out_path=OUTPUT_DIR /
    "embedding_visualization.png"
):

    emb_cols = [
        c
        for c in df_emb.columns
        if c.startswith("emb_")
    ]

    pca = PCA(
        n_components=n_components
    )

    reduced = pca.fit_transform(
        df_emb[emb_cols].values
    )

    plt.figure(
        figsize=(8, 6)
    )

    plt.scatter(
        reduced[:, 0],
        reduced[:, 1],
        s=10,
        alpha=0.7
    )

    plt.title(
        "Transformer User Embedding PCA"
    )

    plt.xlabel("PC1")
    plt.ylabel("PC2")

    plt.tight_layout()

    plt.savefig(
        out_path
    )

    plt.close()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    sequences_file = (
        OUTPUT_DIR /
        "user_sequences.csv"
    )

    if not sequences_file.is_file():

        print(
            "User sequences not found."
        )

        print(
            "Run preprocessing first."
        )

    else:

        model, vocab = train_transformer(
            sequences_file
        )

        df_emb = generate_user_embeddings(
            model,
            vocab,
            sequences_file
        )

        visualize_embeddings(
            df_emb
        )