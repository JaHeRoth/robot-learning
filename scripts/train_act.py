import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from torch.optim import AdamW
from torch.utils.data import DataLoader

from scripts.act import ACT
from scripts.train_common import train_loop


def train(seed: int = 0):
    torch.manual_seed(seed)

    chunk_len = 100
    batch_size = 64
    lr = 2e-5
    weight_decay = 1e-4
    kl_weight = 10.0
    grad_clip_at = 10.0

    num_batches = 100_000

    fps = 10
    ds = LeRobotDataset(
        "lerobot/pusht",
        delta_timestamps={"action": [i / fps for i in range(chunk_len)]}
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=4)

    model = ACT(action_dim=ds.meta.features["action"]["shape"][0], chunk_len=chunk_len).cuda()
    opt = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    state_mean = torch.as_tensor(
        ds.meta.stats["observation.state"]["mean"], dtype=torch.float32, device="cuda"
    )
    state_std = torch.as_tensor(
        ds.meta.stats["observation.state"]["std"], dtype=torch.float32, device="cuda"
    )
    action_mean = torch.as_tensor(
        ds.meta.stats["action"]["mean"], dtype=torch.float32, device="cuda"
    )
    action_std = torch.as_tensor(
        ds.meta.stats["action"]["std"], dtype=torch.float32, device="cuda"
    )

    def loss_fn(model, batch):
        img = batch["observation.image"].unsqueeze(1).cuda()
        proprio = (batch["observation.state"].cuda() - state_mean) / state_std
        chunk = (batch["action"].cuda() - action_mean) / action_std
        loss_mask = ~batch["action_is_pad"].cuda().unsqueeze(-1)

        chunk_pred, z_mean, z_logvar = model(img, proprio, chunk)
        l1_loss = ((chunk_pred - chunk).abs() * loss_mask).mean()
        kl_loss = -(1 + z_logvar - z_mean.pow(2) - z_logvar.exp()).sum(axis=1).mean() / 2
        return l1_loss + kl_weight * kl_loss

    train_loop(
        model=model,
        loader=loader,
        opt=opt,
        loss_fn=loss_fn,
        num_batches=num_batches,
        out_dir="outputs/my_act",
        grad_clip_at=grad_clip_at,
    )


if __name__ == "__main__":
    train()
