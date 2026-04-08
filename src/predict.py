import torch
from src.model import Encoder, Decoder, Seq2Seq

class Translator:
    def __init__(self, model_path, en_vocab, vi_vocab, device='cpu'):
        self.device = torch.device(device)
        self.en_vocab = en_vocab
        self.vi_vocab = vi_vocab
        
        # Tạo lại kiến trúc (Cần khớp với file train)
        encoder = Encoder(len(self.en_vocab), 256, 512).to(self.device)
        decoder = Decoder(len(self.vi_vocab), 256, 512).to(self.device)
        self.model = Seq2Seq(encoder, decoder, self.device).to(self.device)
        
        # Load trọng số đã train
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

    def translate(self, sentence, max_len=70):
        # Biến chữ thành số
        src_indices = self.en_vocab.encode(sentence, max_len)
        src_tensor = torch.tensor(src_indices, dtype=torch.long).unsqueeze(0).to(self.device) # [1, seq_len]
        
        with torch.no_grad():
            encoder_outputs, hidden = self.model.encoder(src_tensor)
            
        # Token bắt đầu
        decoder_input = torch.tensor([[self.vi_vocab.word2idx["<SOS>"]]], dtype=torch.long).to(self.device)
        
        translated_words = []
        for _ in range(max_len):
            with torch.no_grad():
                prediction, hidden, _ = self.model.decoder(decoder_input, hidden, encoder_outputs)
            
            top1 = prediction.argmax(1)
            
            # Nếu gặp <EOS> thì dừng
            if top1.item() == self.vi_vocab.word2idx["<EOS>"]:
                break
                
            translated_words.append(self.vi_vocab.idx2word[top1.item()])
            decoder_input = top1.unsqueeze(1)
            
        return " ".join(translated_words)