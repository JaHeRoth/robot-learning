# %%
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from matplotlib import pyplot as plt
import torch
ds = LeRobotDataset("lerobot/pusht")
print(ds)

# %%
# Q: What does a frame look like?
ds[0]

# %%
# Q: How do states and actions interact?
for i in range(10):
    print(
        "state:",
        ds[i]["observation.state"],
        "action:",
        ds[i]["action"],
    )
# A: State follows action (so action is an absolute target)

# %%
display(ds.features)
display(ds.meta.stats)

# %%
plt.plot(
    torch.stack(ds.hf_dataset["observation.state"])[:100, 0],
    label="State",
)
plt.plot(
    torch.stack(ds.hf_dataset["action"])[:100, 0],
    label="Action",
)
plt.xlabel("Index")
plt.ylabel("x-coordinate")
plt.legend()
plt.grid()
plt.show()
# Confirms that states follow actions as absolute targets

# %%
