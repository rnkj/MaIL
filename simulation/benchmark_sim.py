import logging
import os
import random
import numpy as np
import torch
import wandb
import robosuite
import multiprocessing as mp
from .base_sim import BaseSim
from libero.libero.envs import *
from LIBERO.libero.libero.envs.bddl_utils import get_problem_info
from agents.utils import sim_framework_path
import imgaug.parameters as iap
from imgaug import augmenters as iaa


log = logging.getLogger(__name__)


def assign_process_to_cpu(pid, cpus):
    os.sched_setaffinity(pid, cpus)


def process_image_input(img_tensor):
    # return (img_tensor / 255. - 0.5) * 2.
    return img_tensor / 255.


# aug = iaa.arithmetic.ReplaceElementwise(iap.FromLowerResolution(iap.Binomial(0.02), size_px=8),
#                                         [255])


class MultiTaskSim(BaseSim):
    def __init__(self,
                 num_episode,
                 max_step_per_episode,
                 bddl_file_directory: str,
                 use_eye_in_hand: bool,
                 seed,
                 device,
                 render,
                 n_cores,
                 camera_shape: tuple,
                 data_aug: bool = False,
                 aug_factor: float = 0.02,
                 task_id: int = 0,
                 nms: float = 0.1):
        super().__init__(seed, device, render, n_cores)

        # data augmentation
        self.data_aug = data_aug
        self.aug_factor = aug_factor
        self.aug = iaa.arithmetic.ReplaceElementwise(iap.FromLowerResolution(iap.Binomial(self.aug_factor), size_px=8), [255])

        self.controller_confige = robosuite.load_controller_config(default_controller="OSC_POSE")
        # according to the task_id, load the corresponding bddl file
        self.bddl_file_directory = sim_framework_path(bddl_file_directory)
        bddl_files = os.listdir(self.bddl_file_directory)

        tasks_problem_name = []
        abs_bddl_files = []
        tasks_language = []

        for bddl_file in bddl_files:
            if not bddl_file.endswith(".bddl"):
                continue
            bddl_file_name = os.path.join(self.bddl_file_directory, bddl_file)

            problem_info = get_problem_info(bddl_file_name)

            abs_bddl_files.append(bddl_file_name)
            tasks_problem_name.append(problem_info["problem_name"])
            tasks_language.append(problem_info["language_instruction"])

        self.tasks_problem_name = tasks_problem_name
        self.abs_bddl_files = abs_bddl_files
        self.tasks_language = tasks_language

        self.use_eye_in_hand = use_eye_in_hand
        self.render = render
        self.task_id = task_id

        self.num_episode = num_episode
        self.max_step_per_episode = max_step_per_episode

        self.success_rate = 0

    def eval_agent(self, agent, contexts, context_ind, success, pid, cpu_set):
        print(os.getpid(), cpu_set)
        assign_process_to_cpu(os.getpid(), cpu_set)

        random.seed(pid)
        torch.manual_seed(pid)
        np.random.seed(pid)

        env_ids = []

        print(contexts)

        for i, context in enumerate(contexts):

            if context not in env_ids:
                env_ids.append(context)

                env = TASK_MAPPING[self.tasks_problem_name[context]](
                    self.abs_bddl_files[context],
                    robots=["Panda"],
                    controller_configs=self.controller_confige,
                    gripper_types="default",
                    initialization_noise=None,
                    use_camera_obs=True,
                    has_renderer=True,
                    has_offscreen_renderer=True,
                    render_camera="frontview",
                    render_collision_mesh=False,
                    render_visual_mesh=True,
                    render_gpu_device_id=-1,
                    control_freq=20,
                    horizon=1000,
                    ignore_done=False,
                    hard_reset=True,
                    camera_names=[
                        "agentview",
                        "robot0_eye_in_hand",
                    ],
                    camera_heights=128,
                    camera_widths=128,
                    camera_depths=False,
                    camera_segmentations=None,
                    renderer="mujoco",
                    renderer_config=None,
                )

            agent.reset()
            obs = env.reset()

            # multiprocessing simulation
            for j in range(self.max_step_per_episode):
                agentview_rgb = obs["agentview_image"]

                if self.data_aug:
                    agentview_rgb = self.aug(image=agentview_rgb)

                if self.use_eye_in_hand:
                    eye_in_hand_rgb = obs["robot0_eye_in_hand_image"]
                    state = (agentview_rgb, eye_in_hand_rgb)
                else:
                    state = agentview_rgb

                action = agent.predict(state)[0]
                obs, r, done, _ = env.step(action)

                # if self.render:
                # env.render()

                if r == 1:
                    success[context, context_ind[i]] = r
                    env.close()
                    break

            env.close()

    def test_agent(self, agent, cpu_set=None, epoch=None):
        logging.info("Start testing agent")

        if cpu_set is None:
            num_cpu = mp.cpu_count()
            cpu_set = [i for i in range(num_cpu)]
        else:
            num_cpu = len(cpu_set)

        print("there is {} cpus".format(num_cpu))

        num_tasks = len(self.tasks_language)
        success = torch.zeros([num_tasks, self.num_episode]).share_memory_()
        all_runs = num_tasks * self.num_episode
        ###################################################################
        # distribute every runs on cpu
        ###################################################################
        contexts = np.arange(num_tasks)
        contexts = np.repeat(contexts, self.num_episode)

        context_ind = np.arange(self.num_episode)
        context_ind = np.tile(context_ind, num_tasks)

        repeat_num = all_runs // num_cpu
        repeat_res = all_runs % num_cpu

        workload_array = np.ones([num_cpu], dtype=int)
        workload_array[:repeat_res] += repeat_num
        workload_array[repeat_res:] = repeat_num

        assert np.sum(workload_array) == all_runs

        ind_workload = np.cumsum(workload_array)
        ind_workload = np.concatenate([[0], ind_workload])
        ###################################################################
        ctx = mp.get_context('spawn')
        processes_list = []

        for i in range(self.n_cores):
            p = ctx.Process(target=self.eval_agent,
                            kwargs={
                                "agent": agent,
                                "contexts": contexts[ind_workload[i]:ind_workload[i + 1]],
                                "context_ind": context_ind[ind_workload[i]:ind_workload[i + 1]],
                                "success": success,
                                "pid": i,
                                "cpu_set": set(cpu_set[i:i + 1])
                            },
                            )
            p.start()
            processes_list.append(p)

        [p.join() for p in processes_list]

        success_rate = torch.mean(success, dim=-1)
        average_success = torch.mean(success_rate).item()

        print(f'success array {success.detach()}')

        custom_step = f"{epoch}_custom_step"
        wandb.define_metric(custom_step)
        wandb.define_metric(f"{epoch}_tasks_success", step_metric=custom_step)

        for num in range(num_tasks):
            log.info(f"Task: {self.tasks_language[num][:-27]} {success_rate[num].item()}")

            wandb.log({custom_step: num,
                       f"{epoch}_tasks_success": success_rate[num].item()
                       })

        wandb.log({f"epoch{epoch}_average_success": average_success})
        log.info(f"Average success rate: {average_success}")
