import torch
from torch.utils.data import Dataset, DataLoader


class NMTDataset(Dataset):
    def __init__(self, data_path):
        self.samples = torch.load(data_path)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        return {
            "src_ids": item["src_ids"],
            "tgt_ids": item["tgt_ids"],
            "src_len": item["src_len"],
            "tgt_len": item["tgt_len"]
        }


def collate_fn(batch, pad_idx=0):
    src_batch = [torch.tensor(item["src_ids"], dtype=torch.long) for item in batch]
    tgt_batch = [torch.tensor(item["tgt_ids"], dtype=torch.long) for item in batch]

    src_lengths = torch.tensor([item["src_len"] for item in batch], dtype=torch.long)
    tgt_lengths = torch.tensor([item["tgt_len"] for item in batch], dtype=torch.long)

    src_padded = torch.nn.utils.rnn.pad_sequence(
        src_batch,
        batch_first=True,
        padding_value=pad_idx
    )
    tgt_padded = torch.nn.utils.rnn.pad_sequence(
        tgt_batch,
        batch_first=True,
        padding_value=pad_idx
    )

    return {
        "src": src_padded,
        "tgt": tgt_padded,
        "src_lengths": src_lengths,
        "tgt_lengths": tgt_lengths
    }


def get_dataloader(data_path, batch_size=32, shuffle=True, num_workers=4, pad_idx=0):
    dataset = NMTDataset(data_path)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        collate_fn=lambda batch: collate_fn(batch, pad_idx=pad_idx)
    )
    return dataloader