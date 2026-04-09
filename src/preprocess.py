import os
import random
import torch
import pandas as pd
from collections import Counter


class Vocab:
    def __init__(self, name):
        self.name = name
        self.word2idx = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.idx2word = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.num_words = 4

    def build_vocab(self, sentences, max_vocab_size=30000):
        word_counter = Counter()
        for sentence in sentences:
            words = str(sentence).split()
            word_counter.update(words)

        most_common_words = word_counter.most_common(max_vocab_size - self.num_words)

        for word, _ in most_common_words:
            if word not in self.word2idx:
                self.word2idx[word] = self.num_words
                self.idx2word[self.num_words] = word
                self.num_words += 1

        print(f"Vocab [{self.name}] size: {self.num_words}")

    def encode(self, sentence, max_len):
        words = str(sentence).split()

        idx_list = [self.word2idx["<SOS>"]]
        for word in words:
            idx_list.append(self.word2idx.get(word, self.word2idx["<UNK>"]))
        idx_list.append(self.word2idx["<EOS>"])

        if len(idx_list) > max_len:
            idx_list = idx_list[:max_len - 1] + [self.word2idx["<EOS>"]]

        return idx_list


def split_train_val(df, val_ratio=0.1, seed=42):
    indices = list(range(len(df)))
    rng = random.Random(seed)
    rng.shuffle(indices)

    val_size = int(len(df) * val_ratio)
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    train_df = df.iloc[train_indices].reset_index(drop=True)
    val_df = df.iloc[val_indices].reset_index(drop=True)

    return train_df, val_df


def encode_dataframe(df, src_vocab, tgt_vocab, src_col="en", tgt_col="vi", max_len=70):
    samples = []

    src_sentences = df[src_col].tolist()
    tgt_sentences = df[tgt_col].tolist()

    for src_sentence, tgt_sentence in zip(src_sentences, tgt_sentences):
        src_ids = src_vocab.encode(src_sentence, max_len=max_len)
        tgt_ids = tgt_vocab.encode(tgt_sentence, max_len=max_len)

        samples.append({
            "src_ids": src_ids,
            "tgt_ids": tgt_ids,
            "src_len": len(src_ids),
            "tgt_len": len(tgt_ids)
        })

    return samples


def main():
    raw_dir = "data/raw"
    processed_dir = "data/processed"
    os.makedirs(processed_dir, exist_ok=True)

    raw_train_path = os.path.join(raw_dir, "train.json")

    print("Loading raw train data...")
    df = pd.read_json(raw_train_path)

    max_len = 70
    en_vocab_size = 20000
    vi_vocab_size = 25000
    val_ratio = 0.1
    split_seed = 42

    print("Splitting train/val...")
    train_df, val_df = split_train_val(df, val_ratio=val_ratio, seed=split_seed)

    print(f"Train size: {len(train_df)}")
    print(f"Val size: {len(val_df)}")

    print("Building vocab from train split only...")
    en_vocab = Vocab(name="English")
    en_vocab.build_vocab(train_df["en"].values, max_vocab_size=en_vocab_size)

    vi_vocab = Vocab(name="Vietnamese")
    vi_vocab.build_vocab(train_df["vi"].values, max_vocab_size=vi_vocab_size)

    print("Encoding train and val...")
    train_samples = encode_dataframe(train_df, en_vocab, vi_vocab, max_len=max_len)
    val_samples = encode_dataframe(val_df, en_vocab, vi_vocab, max_len=max_len)

    vocab_data = {
        "src_vocab": {
            "word2idx": en_vocab.word2idx,
            "idx2word": en_vocab.idx2word,
            "num_words": en_vocab.num_words
        },
        "tgt_vocab": {
            "word2idx": vi_vocab.word2idx,
            "idx2word": vi_vocab.idx2word,
            "num_words": vi_vocab.num_words
        },
        "special_tokens": {
            "pad": 0,
            "sos": 1,
            "eos": 2,
            "unk": 3
        },
        "meta": {
            "max_len": max_len,
            "val_ratio": val_ratio,
            "split_seed": split_seed
        }
    }

    print("Saving processed files...")
    torch.save(train_samples, os.path.join(processed_dir, "train.pt"))
    torch.save(val_samples, os.path.join(processed_dir, "val.pt"))
    torch.save(vocab_data, os.path.join(processed_dir, "vocabs.pt"))

    print("Done.")
    print(f"Train samples: {len(train_samples)}")
    print(f"Val samples: {len(val_samples)}")
    print(f"EN vocab size: {en_vocab.num_words}")
    print(f"VI vocab size: {vi_vocab.num_words}")


if __name__ == "__main__":
    main()