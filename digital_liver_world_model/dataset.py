import torch
from torch.utils.data import Dataset, DataLoader


class LiverDataset(Dataset):
    def __init__(
        self,
        trajectories: torch.Tensor,
        contexts: list[dict],
        context_tensors: torch.Tensor | None = None,
        context_length: int = 24,
        forecast_horizon: int = 12,
        stride: int = 1,
    ):
        self.trajectories = trajectories
        self.contexts = contexts
        self.context_tensors = context_tensors
        self.ctx = context_length
        self.fc = forecast_horizon

        n_steps = trajectories.shape[1]

        self.treatment_masks = []
        self.ercp_masks = []
        for ctx in contexts:
            tm = torch.zeros(n_steps)
            tm[ctx["udca_start"]:] = 1.0
            self.treatment_masks.append(tm)

            em = torch.zeros(n_steps)
            for m in ctx["ercp_months"]:
                em[m] = 1.0
            self.ercp_masks.append(em)

        self.sample_indices: list[tuple[int, int]] = []
        max_start = n_steps - context_length - forecast_horizon
        for i in range(len(trajectories)):
            for start in range(0, max_start + 1, stride):
                self.sample_indices.append((i, start))

    def __len__(self) -> int:
        return len(self.sample_indices)

    def __getitem__(self, idx: int) -> dict:
        p, start = self.sample_indices[idx]
        end = start + self.ctx
        item = {
            "context": self.trajectories[p, start:end],
            "target": self.trajectories[p, end:end + self.fc],
            "treatment": self.treatment_masks[p][start:end + self.fc],
            "ercp": self.ercp_masks[p][start:end + self.fc],
            "context_time": torch.arange(start, end).float(),
            "target_time": torch.arange(end, end + self.fc).float(),
        }
        if self.context_tensors is not None:
            item["context_tensor"] = self.context_tensors[p]
        return item


def make_loaders(
    trajectories: torch.Tensor,
    contexts: list[dict],
    context_tensors: torch.Tensor | None = None,
    batch_size: int = 64,
    train_frac: float = 0.8,
    test_frac: float = 0.1,
    context_length: int = 24,
    forecast_horizon: int = 12,
    stride: int = 1,
    seed: int = 42,
    pin_memory: bool = False,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, DataLoader, "LiverDataset", "LiverDataset", "LiverDataset"]:
    n = len(trajectories)
    n_train = int(n * train_frac)
    n_test = int(n * test_frac)
    n_val = n - n_train - n_test
    rng = torch.Generator().manual_seed(seed)
    idx = torch.randperm(n, generator=rng)
    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train + n_val]
    test_idx = idx[n_train + n_val:]

    if context_tensors is not None:
        ctx_train = context_tensors[train_idx]
        ctx_val = context_tensors[val_idx]
        ctx_test = context_tensors[test_idx]
    else:
        ctx_train = ctx_val = ctx_test = None

    train_ds = LiverDataset(
        trajectories[train_idx], [contexts[int(i)] for i in train_idx],
        ctx_train, context_length, forecast_horizon, stride,
    )
    val_ds = LiverDataset(
        trajectories[val_idx], [contexts[int(i)] for i in val_idx],
        ctx_val, context_length, forecast_horizon,
    )
    test_ds = LiverDataset(
        trajectories[test_idx], [contexts[int(i)] for i in test_idx],
        ctx_test, context_length, forecast_horizon,
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              pin_memory=pin_memory, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size,
                            pin_memory=pin_memory, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size,
                             pin_memory=pin_memory, num_workers=num_workers)
    return train_loader, val_loader, test_loader, train_ds, val_ds, test_ds
