import torch
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import re

class Vocab:
    def __init__(self, name):
        self.name = name
        # 4 Token đặc biệt bắt buộc phải có trong dịch máy
        self.word2idx = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.idx2word = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.num_words = 4

    def build_vocab(self, sentences, max_vocab_size=30000):
        """
        Quét qua toàn bộ câu để đếm tần suất và giữ lại các từ phổ biến nhất.
        """
        word_counter = Counter()
        for sentence in sentences:
            # Tách từ đơn giản bằng khoảng trắng (vì bạn đã dùng underthesea ở notebook)
            words = str(sentence).split()
            word_counter.update(words)
        
        # Lấy ra những từ xuất hiện nhiều nhất, giới hạn bằng max_vocab_size
        most_common_words = word_counter.most_common(max_vocab_size - self.num_words)
        
        for word, _ in most_common_words:
            if word not in self.word2idx:
                self.word2idx[word] = self.num_words
                self.idx2word[self.num_words] = word
                self.num_words += 1
                
        print(f"Từ điển [{self.name}] đã được xây dựng với {self.num_words} từ.")

    def encode(self, sentence, max_len, pad=False):
        words = str(sentence).split()
        idx_list = [self.word2idx["<SOS>"]]

        for word in words:
            idx_list.append(self.word2idx.get(word, self.word2idx["<UNK>"]))

        idx_list.append(self.word2idx["<EOS>"])

        if len(idx_list) > max_len:
            idx_list = idx_list[:max_len-1] + [self.word2idx["<EOS>"]]

        if pad:
            padding_len = max_len - len(idx_list)
            if padding_len > 0:
                idx_list.extend([self.word2idx["<PAD>"]] * padding_len)

        return idx_list

    def decode(self, idx_list):
        """
        Biến một list các ID ngược lại thành câu chữ (dùng khi xem kết quả dịch).
        """
        return [self.idx2word.get(idx, "<UNK>") for idx in idx_list]


class NMTDataset(Dataset):
    def __init__(self, df, src_vocab, tgt_vocab, src_col='en', tgt_col='vi', max_len=70):
        self.samples = []

        src_sentences = df[src_col].tolist()
        tgt_sentences = df[tgt_col].tolist()

        for src_sentence, tgt_sentence in zip(src_sentences, tgt_sentences):
            src_ids = src_vocab.encode(src_sentence, max_len=max_len, pad=False)
            tgt_ids = tgt_vocab.encode(tgt_sentence, max_len=max_len, pad=False)
            self.samples.append((src_ids, tgt_ids))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        src_ids, tgt_ids = self.samples[idx]
        return {
            "src": src_ids,
            "tgt": tgt_ids
        }


def get_dataloader(df, src_vocab, tgt_vocab, batch_size=32, max_len=70, shuffle=True):
    """
    Hàm tạo DataLoader để chia nhỏ dữ liệu thành các batch khi train.
    """
    dataset = NMTDataset(df, src_vocab, tgt_vocab, src_col='en', tgt_col='vi', max_len=max_len)
    
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle,
        num_workers=2 # Tăng tốc độ đọc dữ liệu bằng đa luồng
    )
    return dataloader