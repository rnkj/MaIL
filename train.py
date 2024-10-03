import os
import logging
import random

import hydra
import numpy as np
import multiprocessing as mp
import wandb
from omegaconf import DictConfig, OmegaConf
import torch

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


@hydra.main(config_path="config", config_name="benchmark_libero_goal.yaml")
def main(cfg: DictConfig) -> None:

    set_seed_everywhere(cfg.seed)

    # init wandb logger and config from hydra path
    wandb.config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)

    run = wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        group=cfg.group,
        mode="offline",
        config=wandb.config
    )

    agent = hydra.utils.instantiate(cfg.agents)
    # train the agents
    agent.train_vision_agent()

    # load the model performs best on the evaluation set
    # agent.load_pretrained_model('/home/i53/student/wang/OCIL/OCIL/libero_10/agent', sv_name="last_ddpm.pth")

    num_cpu = mp.cpu_count()
    cpu_set = list(range(num_cpu))
    print("there are cpus: ", num_cpu)

    assign_cpus = cpu_set[cfg.seed * cfg.n_cores:cfg.seed * cfg.n_cores + cfg.n_cores]

    env_sim = hydra.utils.instantiate(cfg.simulation)
    env_sim.test_agent(agent, assign_cpus, epoch=agent.epoch)

    log.info("done")

    wandb.finish()


if __name__ == "__main__":
    main()