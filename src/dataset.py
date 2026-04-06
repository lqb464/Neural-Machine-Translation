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

    def encode(self, sentence, max_len):
        """
        Biến một câu (chuỗi chữ) thành một list các ID (số nguyên) và thực hiện Padding.
        """
        words = str(sentence).split()
        
        # Khởi tạo list chứa ID, bắt đầu bằng <SOS>
        idx_list = [self.word2idx["<SOS>"]]
        
        # Thêm ID của các từ trong câu (nếu từ lạ thì dùng <UNK>)
        for word in words:
            idx_list.append(self.word2idx.get(word, self.word2idx["<UNK>"]))
            
        # Thêm <EOS> báo hiệu kết thúc câu
        idx_list.append(self.word2idx["<EOS>"])
        
        # Cắt bớt nếu câu quá dài
        if len(idx_list) > max_len:
            idx_list = idx_list[:max_len-1] + [self.word2idx["<EOS>"]]
            
        # Padding bằng <PAD> (ID = 0) nếu câu quá ngắn
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
        """
        Dataset quản lý việc nạp dữ liệu từ DataFrame vào PyTorch.
        """
        self.df = df
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.src_col = src_col
        self.tgt_col = tgt_col
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Lấy câu ở hàng thứ idx
        src_sentence = self.df.iloc[idx][self.src_col]
        tgt_sentence = self.df.iloc[idx][self.tgt_col]
        
        # Biến chữ thành số
        src_indices = self.src_vocab.encode(src_sentence, self.max_len)
        tgt_indices = self.tgt_vocab.encode(tgt_sentence, self.max_len)
        
        # Chuyển thành Tensor kiểu Long (số nguyên) để nạp vào mô hình
        return {
            'src': torch.tensor(src_indices, dtype=torch.long),
            'tgt': torch.tensor(tgt_indices, dtype=torch.long)
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