import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import wandb
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

    # 2. Chuẩn bị Data
    print("Đang nạp dữ liệu...")
    df = pd.read_json("data/raw/train.json") 
    
    en_vocab = Vocab(name="English")
    en_vocab.build_vocab(df['en'].values, max_vocab_size=config.en_vocab_size)
    
    vi_vocab = Vocab(name="Vietnamese")
    vi_vocab.build_vocab(df['vi'].values, max_vocab_size=config.vi_vocab_size)

    dataloader = get_dataloader(df, en_vocab, vi_vocab, batch_size=config.batch_size, max_len=config.max_len)

    # 3. Khởi tạo Mô hình
    encoder = Encoder(config.en_vocab_size, config.embed_dim, config.hidden_dim).to(DEVICE)
    decoder = Decoder(config.vi_vocab_size, config.embed_dim, config.hidden_dim).to(DEVICE)
    model = Seq2Seq(encoder, decoder, DEVICE).to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    # 4. Loop Training
    print("Bắt đầu huấn luyện...")
    for epoch in range(config.epochs):
        model.train()
        total_loss = 0
        
        for batch_idx, batch in enumerate(dataloader):
            src = batch['src'].to(DEVICE)
            tgt = batch['tgt'].to(DEVICE)

            optimizer.zero_grad()
            output = model(src, tgt)

            output_dim = output.shape[-1]
            output_flatten = output[1:].view(-1, output_dim)
            tgt_flatten = tgt[:, 1:].contiguous().view(-1)

            loss = criterion(output_flatten, tgt_flatten)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # Log loss của từng batch lên W&B
            wandb.log({"batch_loss": loss.item()})
            
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{config.epochs}] - Loss: {avg_loss:.4f}")
        wandb.log({"epoch_loss": avg_loss, "epoch": epoch + 1})
        
        # Lưu trọng số
        os.makedirs("weights", exist_ok=True)
        torch.save(model.state_dict(), f"weights/model_epoch_{epoch+1}.pth")

    import pickle
    print("Đang lưu trữ từ điển...")
    with open("weights/en_vocab.pkl", "wb") as f:
        pickle.dump(en_vocab, f)
    with open("weights/vi_vocab.pkl", "wb") as f:
        pickle.dump(vi_vocab, f)
    print("Đã lưu từ điển thành công!")
        
    wandb.finish()

if __name__ == "__main__":
    train()