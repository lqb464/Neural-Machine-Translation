import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import wandb
import pickle
from tqdm import tqdm
from src.dataset import Vocab, get_dataloader
from src.model import Encoder, Decoder, Seq2Seq

def train():
    # 1. Khởi tạo W&B
    wandb.init(
        project="neural-machine-translation",
        config={
            "learning_rate": 0.001,
            "epochs": 10,
            "batch_size": 64,
            "embed_dim": 256,
            "hidden_dim": 512,
            "max_len": 70,
            "en_vocab_size": 20000,
            "vi_vocab_size": 25000
        }
    )
    config = wandb.config
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Đang huấn luyện trên thiết bị: {DEVICE}")

    # 2. Chuẩn bị Data & Vocab
    print("Đang nạp dữ liệu và xây dựng từ điển...")
    df = pd.read_json("/kaggle/input/datasets/lqb464/dataset/train.json") 
    
    en_vocab = Vocab(name="English")
    en_vocab.build_vocab(df['en'].values, max_vocab_size=config.en_vocab_size)
    
    vi_vocab = Vocab(name="Vietnamese")
    vi_vocab.build_vocab(df['vi'].values, max_vocab_size=config.vi_vocab_size)

    # Lưu từ điển NGAY để đảm bảo tính nhất quán
    os.makedirs("weights", exist_ok=True)
    with open("weights/en_vocab.pkl", "wb") as f:
        pickle.dump(en_vocab, f)
    with open("weights/vi_vocab.pkl", "wb") as f:
        pickle.dump(vi_vocab, f)

    dataloader = get_dataloader(df, en_vocab, vi_vocab, batch_size=config.batch_size, max_len=config.max_len)

    # 3. Khởi tạo Mô hình
    encoder = Encoder(config.en_vocab_size, config.embed_dim, config.hidden_dim).to(DEVICE)
    decoder = Decoder(config.vi_vocab_size, config.embed_dim, config.hidden_dim).to(DEVICE)
    model = Seq2Seq(encoder, decoder, DEVICE).to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    # PAD_IDX thường là 0, ignore_index giúp mô hình không học cách dự đoán các từ đệm
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    # 4. Loop Training
    print("Bắt đầu huấn luyện...")
    for epoch in range(config.epochs):
        model.train()
        total_loss = 0
        
        # Thêm tqdm để theo dõi tiến độ ngay tại Terminal
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config.epochs}")
        
        for batch in progress_bar:
            src = batch['src'].to(DEVICE)
            tgt = batch['tgt'].to(DEVICE)

            optimizer.zero_grad()
            
            # Forward pass
            output = model(src, tgt) # Output shape: [batch, seq_len, vocab_size]

            # --- TÍNH LOSS (PHẦN QUAN TRỌNG NHẤT) ---
            output_dim = output.shape[-1]
            
            # Nếu batch_first=True:
            # output[:, 1:] bỏ qua dự đoán cho token SOS
            # tgt[:, 1:] bỏ qua token SOS trong label thực tế
            output_flatten = output[:, 1:, :].reshape(-1, output_dim)
            tgt_flatten = tgt[:, 1:].reshape(-1)

            loss = criterion(output_flatten, tgt_flatten)
            
            loss.backward()
            # Clip gradient để tránh nổ gradient (Exploding Gradients)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})
            wandb.log({"batch_loss": loss.item()})
            
        avg_loss = total_loss / len(dataloader)
        print(f"Kết thúc Epoch {epoch+1} - Average Loss: {avg_loss:.4f}")
        wandb.log({"epoch_loss": avg_loss, "epoch": epoch + 1})
        
        # Lưu trọng số sau mỗi Epoch
        torch.save(model.state_dict(), f"weights/model_epoch_{epoch+1}.pth")

    print("Huấn luyện hoàn tất! Từ điển và trọng số đã sẵn sàng tại thư mục weights/.")
    wandb.finish()

if __name__ == "__main__":
    train()