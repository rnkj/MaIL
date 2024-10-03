import os
import logging
import random

import hydra
import numpy as np
import multiprocessing as mp
import wandb
from omegaconf import DictConfig, OmegaConf
import torch
import time

from agents.utils import sim_framework_path

log = logging.getLogger(__name__)

print(torch.cuda.is_available())

OmegaConf.register_new_resolver(
    "add", lambda *numbers: sum(numbers)
)
torch.cuda.empty_cache()


def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


@hydra.main(config_path="config", config_name="multi_task.yaml")
def main(cfg: DictConfig) -> None:

    set_seed_everywhere(cfg.seed)

    wandb.config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)

    run = wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        group=cfg.group,
        mode="disabled",
        config=wandb.config
    )

    agent = hydra.utils.instantiate(cfg.agents)

    all_time = []

    for data in agent.train_dataloader:
        bp_imgs, inhand_imgs, action, mask = data

        bp_imgs = bp_imgs.to(agent.device)
        inhand_imgs = inhand_imgs.to(agent.device)

        # obs = agent.scaler.scale_input(obs)
        action = agent.scaler.scale_output(action)

        action = action[:, agent.obs_seq_len - 1:, :].contiguous()

        # obs = obs[:, :agent.obs_seq_len].contiguous()
        bp_imgs = bp_imgs[:, :agent.obs_seq_len].contiguous()
        inhand_imgs = inhand_imgs[:, :agent.obs_seq_len].contiguous()

        state = (bp_imgs, inhand_imgs)

        start = time.time()

        model_pred = agent.model(state, goal=None)

        end = time.time()

        all_time.append(end - start)

    all_time = np.array(all_time)
    print('mean inference time is: ', all_time.mean())


if __name__ == "__main__":
    main()
