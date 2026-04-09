import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from tqdm import tqdm

from src.dataset import get_dataloader
from src.model import Encoder, Decoder, Seq2Seq


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0

    for batch in dataloader:
        src = batch["src"].to(device, non_blocking=True)
        tgt = batch["tgt"].to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            output = model(src, tgt, teacher_forcing_ratio=0.0)
            output_dim = output.shape[-1]

            output_flatten = output[:, 1:, :].reshape(-1, output_dim)
            tgt_flatten = tgt[:, 1:].reshape(-1)

            loss = criterion(output_flatten, tgt_flatten)

        total_loss += loss.item()

    return total_loss / len(dataloader)


def train():
    wandb.init(
        project="neural-machine-translation",
        config={
            "learning_rate": 0.001,
            "epochs": 10,
            "batch_size": 64,
            "embed_dim": 256,
            "hidden_dim": 512,
            "num_workers": 4,
            "teacher_forcing_ratio": 0.5,
            "log_every": 100
        }
    )

    config = wandb.config
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    processed_dir = "data/processed"
    train_path = os.path.join(processed_dir, "train.pt")
    val_path = os.path.join(processed_dir, "val.pt")
    vocab_path = os.path.join(processed_dir, "vocabs.pt")

    print("Loading vocab...")
    vocab_data = torch.load(vocab_path)
    en_vocab_size = vocab_data["src_vocab"]["num_words"]
    vi_vocab_size = vocab_data["tgt_vocab"]["num_words"]
    pad_idx = vocab_data["special_tokens"]["pad"]

    print("Building dataloaders...")
    train_loader = get_dataloader(
        train_path,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pad_idx=pad_idx
    )

    val_loader = get_dataloader(
        val_path,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pad_idx=pad_idx
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    encoder = Encoder(en_vocab_size, config.embed_dim, config.hidden_dim).to(device)
    decoder = Decoder(vi_vocab_size, config.embed_dim, config.hidden_dim).to(device)
    model = Seq2Seq(encoder, decoder, device).to(device)

    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    os.makedirs("weights", exist_ok=True)

    best_val_loss = float("inf")
    global_step = 0

    for epoch in range(config.epochs):
        model.train()
        total_train_loss = 0.0
        epoch_start = time.time()

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config.epochs}")

        for batch in progress_bar:
            src = batch["src"].to(device, non_blocking=True)
            tgt = batch["tgt"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                output = model(src, tgt, teacher_forcing_ratio=config.teacher_forcing_ratio)
                output_dim = output.shape[-1]

                output_flatten = output[:, 1:, :].reshape(-1, output_dim)
                tgt_flatten = tgt[:, 1:].reshape(-1)

                loss = criterion(output_flatten, tgt_flatten)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            total_train_loss += loss.item()
            global_step += 1

            progress_bar.set_postfix({"train_loss": f"{loss.item():.4f}"})

            if global_step % config.log_every == 0:
                wandb.log({
                    "batch_train_loss": loss.item(),
                    "global_step": global_step
                })

        avg_train_loss = total_train_loss / len(train_loader)
        val_loss = evaluate(model, val_loader, criterion, device)
        epoch_time = time.time() - epoch_start

        print(
            f"Epoch {epoch + 1} done | "
            f"train_loss={avg_train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"time={epoch_time:.2f}s"
        )

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": val_loss,
            "epoch_time_sec": epoch_time
        })

        torch.save(model.state_dict(), f"weights/model_epoch_{epoch + 1}.pth")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "weights/best_model.pth")
            print(f"Saved new best model with val_loss={best_val_loss:.4f}")

    print("Training finished.")
    wandb.finish()


if __name__ == "__main__":
    train()