"""lerobot-eval with per-request HTTP logging and known-benign warnings silenced.

httpx logs every Hub request at INFO; cap it (and friends) at WARNING
before lerobot configures logging. The gymnasium wrapper-passthrough
deprecation and lerobot's task_description notice fire every batch and
don't apply to PushT. Same args as lerobot-eval.
"""

import logging
import warnings

for _name in ("httpx", "httpcore", "urllib3", "huggingface_hub"):
    logging.getLogger(_name).setLevel(logging.WARNING)

warnings.filterwarnings("ignore", message=r"WARN: env\.task")
warnings.filterwarnings("ignore", message=r"The environment does not have 'task_description'")

from lerobot.scripts.eval import main

if __name__ == "__main__":
    main()
