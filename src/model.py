import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_layers=1):
        super(Encoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        
        # 1. Lớp Embedding: Biến ID của từ thành vector liên tục
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # 2. Lớp GRU: Xử lý chuỗi
        # batch_first=True nghĩa là dữ liệu đầu vào có dạng [Batch_size, Seq_len]
        self.gru = nn.GRU(embed_dim, hidden_dim, n_layers, batch_first=True)
        
    def forward(self, x):
        # x: [batch_size, seq_len]
        
        # Đi qua embedding -> [batch_size, seq_len, embed_dim]
        embedded = self.embedding(x)
        
        # Đi qua GRU
        # outputs: Chứa hidden states của TẤT CẢ các bước thời gian (Cần cho Attention)
        # hidden: Hidden state của bước CUỐI CÙNG (Cần để mồi cho Decoder)
        outputs, hidden = self.gru(embedded)
        
        return outputs, hidden


class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_layers=1):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # GRU của Decoder nhận vào cả vector embedding và vector context từ Attention
        self.gru = nn.GRU(embed_dim + hidden_dim, hidden_dim, n_layers, batch_first=True)
        
        # Attention
        self.attention = LuongAttention(hidden_dim)
        
        # Lớp Linear để dự đoán từ vựng trong từ điển tiếng Việt
        self.out = nn.Linear(hidden_dim * 2, vocab_size)
        
    def forward(self, x, hidden, encoder_outputs):
        # x: [batch_size, 1] -> ID của 1 từ ở bước hiện tại
        # hidden: [1, batch_size, hidden_dim]
        # encoder_outputs: [batch_size, seq_len, hidden_dim]
        
        # 1. Đi qua embedding -> [batch_size, 1, embed_dim]
        embedded = self.embedding(x)
        
        # 2. Tính attention để lấy vector ngữ cảnh (context)
        context, attn_weights = self.attention(hidden, encoder_outputs)
        
        # 3. Gộp (concatenate) embedding và context lại
        # Kết quả: [batch_size, 1, embed_dim + hidden_dim]
        rnn_input = torch.cat((embedded, context), dim=2)
        
        # 4. Đi qua GRU
        output, hidden = self.gru(rnn_input, hidden)
        
        # 5. Gộp output của GRU và context để đưa qua lớp Linear cuối cùng
        output = torch.cat((output, context), dim=2)
        
        # Kết quả: [batch_size, vocab_size] -> Phân phối xác suất của từ tiếp theo
        prediction = self.out(output.squeeze(1))
        
        return prediction, hidden, attn_weights
    

class LuongAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(LuongAttention, self).__init__()
        # Ma trận trọng số Wa trong công thức General Attention
        self.wa = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, decoder_hidden, encoder_outputs):
        # decoder_hidden: [1, batch_size, hidden_dim] -> Trạng thái hiện tại của Decoder
        # encoder_outputs: [batch_size, seq_len, hidden_dim] -> Toàn bộ từ của Encoder
        
        # Đưa decoder_hidden về dạng [batch_size, hidden_dim, 1] để nhân ma trận
        decoder_hidden_permuted = decoder_hidden.permute(1, 2, 0)
        
        # Bước 1: Nhân encoder_outputs với ma trận Wa
        # Kết quả: [batch_size, seq_len, hidden_dim]
        score = self.wa(encoder_outputs)
        
        # Bước 2: Nhân vô hướng với decoder_hidden để tính điểm tương quan
        # Kết quả: [batch_size, seq_len, 1]
        scores = torch.bmm(score, decoder_hidden_permuted)
        
        # Bước 3: Đưa qua hàm Softmax để tạo thành phân phối xác suất (tổng = 1)
        # Kết quả: [batch_size, seq_len, 1]
        attention_weights = F.softmax(scores, dim=1)
        
        # Bước 4: Tính vector ngữ cảnh (Context Vector) bằng cách nhân trọng số với encoder_outputs
        # Kết quả: [batch_size, 1, hidden_dim]
        context = torch.bmm(attention_weights.permute(0, 2, 1), encoder_outputs)
        
        return context, attention_weights
    

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
            
            # SỬA TẠI ĐÂY: Khởi tạo với batch_size ở chiều đầu tiên
            # [batch_size, seq_len, vocab_size]
            outputs = torch.zeros(batch_size, max_len, tgt_vocab_size).to(self.device)
            
            encoder_outputs, hidden = self.encoder(src)
            decoder_input = tgt[:, 0].unsqueeze(1) 
            
            for t in range(1, max_len):
                prediction, hidden, _ = self.decoder(decoder_input, hidden, encoder_outputs)
                
                # SỬA TẠI ĐÂY: Lưu dự đoán vào đúng vị trí của batch_first
                outputs[:, t, :] = prediction
                
                top1 = prediction.argmax(1) 
                use_teacher_forcing = random.random() < teacher_forcing_ratio
                decoder_input = tgt[:, t].unsqueeze(1) if use_teacher_forcing else top1.unsqueeze(1)
                
            return outputs # Trả về [batch_size, seq_len, vocab_size]