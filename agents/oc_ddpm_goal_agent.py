from collections import deque
import os
import logging
from typing import Optional

from omegaconf import DictConfig
import hydra
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import wandb
import einops
# from sklearn.neighbors import KernelDensity
# import loralib as lora

from agents.base_agent import BaseAgent
from agents.models.oc_ddpm.ema import ExponentialMovingAverage

# A logger for this file
log = logging.getLogger(__name__)


class DiffusionPolicy(nn.Module):
    def __init__(self, model: DictConfig, obs_encoder: DictConfig,
                 visual_input: bool = True, device: str = "cpu"):
        super(DiffusionPolicy, self).__init__()

        self.visual_input = visual_input

        self.obs_encoder = hydra.utils.instantiate(obs_encoder).to(device)
        # self.goal_encoder = hydra.utils.instantiate(goal_encoder).to(device)

        self.model = hydra.utils.instantiate(model).to(device)

        self.linear = nn.Linear(512, 256)

    def forward(self, inputs, goal, action=None, if_train=False, if_return_obs=False):
        # encode state and visual inputs
        # the encoder should be shared by all the baselines

        if self.visual_input:
            agentview_image, in_hand_image, goal_imgs = inputs

            B, T, C, H, W = agentview_image.size()

            agentview_image = agentview_image.view(B * T, C, H, W)
            in_hand_image = in_hand_image.view(B * T, C, H, W)
            # state = state.view(B * T, -1)

            # bp_imgs = einops.rearrange(bp_imgs, "B T C H W -> (B T) C H W")
            # inhand_imgs = einops.rearrange(inhand_imgs, "B T C H W -> (B T) C H W")
            # _, T1, C1, H1, W1 = goal_imgs.size()
            # goal_imgs = goal_imgs.view(B * T1, C1, H1, W1)
            #
            # goals_dict = {"goal_rgb": goal_imgs}
            # goal_emb = self.goal_encoder(goals_dict)

            goal_emb = self.linear(goal_imgs)

            obs_dict = {"agentview_rgb": agentview_image,
                        "eye_in_hand_rgb": in_hand_image,}
                        # "robot_ee_pos": state}

            obs = self.obs_encoder(obs_dict)
            obs = obs.view(B, T, -1)

            # goal_emb = goal_emb.view(B, T1, -1)

            obs = torch.cat((goal_emb, obs), dim=1)

        else:
            obs = self.obs_encoder(inputs)

        if if_train:
            return self.model.loss(action, obs, goal)

        # make prediction
        pred = self.model(obs, goal)

        if if_return_obs:
            return pred, obs

        return pred

    def get_params(self):
        return self.parameters()


class DiffusionAgent(BaseAgent):

    def __init__(
            self,
            model: DictConfig,
            optimization: DictConfig,
            trainset: DictConfig,
            valset: DictConfig,
            train_batch_size,
            val_batch_size,
            num_workers,
            device: str,
            epoch: int,
            scale_data,
            use_ema: bool,
            discount: int,
            decay: float,
            update_ema_every_n_steps: int,
            goal_window_size: int,
            window_size: int,
            obs_seq_len: int,
            action_seq_size: int,
            pred_last_action_only: bool = False,
            diffusion_kde: bool = False,
            diffusion_kde_samples: int = 100,
            goal_conditioned: bool = False,
            eval_every_n_epochs: int = 50
    ):
        super().__init__(model, trainset=trainset, valset=valset, train_batch_size=train_batch_size,
                         val_batch_size=val_batch_size, num_workers=num_workers, device=device,
                         epoch=epoch, scale_data=scale_data, eval_every_n_epochs=eval_every_n_epochs)

        # Define the bounds for the sampler class
        self.model.model.min_action = torch.from_numpy(self.scaler.y_bounds[0, :]).to(self.device)
        self.model.model.max_action = torch.from_numpy(self.scaler.y_bounds[1, :]).to(self.device)

        # lora.mark_only_lora_as_trainable(self.model)

        self.eval_model_name = "eval_best_ddpm.pth"
        self.last_model_name = "last_ddpm.pth"

        self.optimizer = hydra.utils.instantiate(
            optimization, params=self.model.parameters()
        )

        self.steps = 0

        self.ema_helper = ExponentialMovingAverage(self.model.parameters(), decay, self.device)
        self.use_ema = use_ema
        self.discount = discount
        self.decay = decay
        self.update_ema_every_n_steps = update_ema_every_n_steps
        # here all the parameters required for the GPT variant
        self.goal_window_size = goal_window_size
        self.window_size = window_size
        self.pred_last_action_only = pred_last_action_only

        self.goal_condition = goal_conditioned

        # if we use DiffusionGPT we need an action context
        if not self.pred_last_action_only:
            self.action_context = deque(maxlen=self.window_size - 1)
            self.que_actions = True
        else:
            self.que_actions = False

        self.diffusion_kde = diffusion_kde
        self.diffusion_kde_samples = diffusion_kde_samples

        self.obs_seq_len = obs_seq_len
        self.action_seq_size = action_seq_size
        self.action_counter = self.action_seq_size

        self.bp_image_context = deque(maxlen=self.obs_seq_len)
        self.inhand_image_context = deque(maxlen=self.obs_seq_len)
        self.des_robot_pos_context = deque(maxlen=self.window_size)

        self.obs_context = deque(maxlen=self.obs_seq_len)
        self.goal_context = deque(maxlen=self.goal_window_size)

    def train_agent(self):

        best_test_mse = 1e10

        for num_epoch in tqdm(range(self.epoch)):

            # run a test batch every n steps
            if not (num_epoch+1) % self.eval_every_n_epochs:

                test_mse = []
                for data in self.test_dataloader:

                    if self.goal_condition:
                        state, action, mask, goal = data
                        mean_mse = self.evaluate(state, action, goal)
                    else:
                        state, action, mask = data
                        mean_mse = self.evaluate(state, action)

                    test_mse.append(mean_mse)
                avrg_test_mse = sum(test_mse) / len(test_mse)

                log.info("Epoch {}: Mean test mse is {}".format(num_epoch, avrg_test_mse))
                if avrg_test_mse < best_test_mse:
                    best_test_mse = avrg_test_mse
                    self.store_model_weights(self.working_dir, sv_name=self.eval_model_name)

                    wandb.log(
                        {
                            "best_model_epochs": num_epoch
                        }
                    )

                    log.info('New best test loss. Stored weights have been updated!')

            train_loss = []
            for data in self.train_dataloader:

                if self.goal_condition:
                    state, action, mask, goal = data
                    batch_loss = self.train_step(state, action, goal)
                else:
                    state, action, mask = data
                    batch_loss = self.train_step(state, action)

                train_loss.append(batch_loss)

                wandb.log(
                    {
                        "loss": batch_loss,
                    }
                )

        self.store_model_weights(self.working_dir, sv_name=self.last_model_name)

        log.info("Training done!")

    def train_vision_agent(self):

        # for num_epoch in tqdm(range(self.epoch)):
        #
        #     train_loss = []

        for data in self.train_dataloader:
            bp_imgs, inhand_imgs, action, goal_imgs = data

            bp_imgs = bp_imgs.to(self.device)
            inhand_imgs = inhand_imgs.to(self.device)

            goal_imgs = goal_imgs.to(self.device)

            # obs = self.scaler.scale_input(obs)
            action = self.scaler.scale_output(action)

            action = action[:, self.obs_seq_len - 1:, :].contiguous()

            # obs = obs[:, :self.obs_seq_len].contiguous()
            bp_imgs = bp_imgs[:, :self.obs_seq_len].contiguous()
            inhand_imgs = inhand_imgs[:, :self.obs_seq_len].contiguous()

            state = (bp_imgs, inhand_imgs, goal_imgs)

            batch_loss = self.train_step(state, action)

                # train_loss.append(batch_loss)

            wandb.log({"train_loss": batch_loss.item()})

            # log.info("Epoch {}: Mean train loss is {}".format(num_epoch, batch_loss.item()))

        # log.info("training done")
        # self.store_model_weights(self.working_dir, sv_name=self.last_model_name)

    def train_step(self, state: torch.Tensor, action: torch.Tensor, goal: Optional[torch.Tensor] = None) -> float:

        # state = state.to(self.device).to(torch.float32)  # [B, V]
        # action = action.to(self.device).to(torch.float32)  # [B, D]
        # scale data if necessarry, otherwise the scaler will return unchanged values
        self.model.train()

        # state = self.scaler.scale_input(state)
        # action = self.scaler.scale_output(action)

        if goal is not None:
            goal = self.scaler.scale_input(goal)

        # action = action[:, self.obs_seq_len-1:, :]
        # state = state[:, :self.obs_seq_len, :]

        # Compute the loss.
        loss = self.model(state, goal, action=action, if_train=True)
        # Before the backward pass, zero all the network gradients
        self.optimizer.zero_grad()
        # Backward pass: compute gradient of the loss with respect to parameters
        loss.backward()
        # Calling the step function to update the parameters
        self.optimizer.step()

        self.steps += 1

        # update the ema model
        if self.steps % self.update_ema_every_n_steps == 0:
            self.ema_helper.update(self.model.parameters())
        return loss

    @torch.no_grad()
    def evaluate(
            self, state: torch.tensor, action: torch.tensor, goal: Optional[torch.Tensor] = None
    ) -> float:

        # scale data if necessarry, otherwise the scaler will return unchanged values
        state = self.scaler.scale_input(state)
        action = self.scaler.scale_output(action)

        action = action[:, self.obs_seq_len - 1:, :]
        state = state[:, :self.obs_seq_len, :]

        if goal is not None:
            goal = self.scaler.scale_input(goal)

        total_mse = 0.0
        # use the EMA model variant
        if self.use_ema:
            self.ema_helper.store(self.model.parameters())
            self.ema_helper.copy_to(self.model.parameters())

        self.model.eval()

        # Compute the loss.
        loss = self.model.loss(action, state, goal)

        # model_pred = self.model(state, goal)
        # mse = nn.functional.mse_loss(model_pred, action, reduction="none")

        total_mse += loss.mean().item()

        # restore the previous model parameters
        if self.use_ema:
            self.ema_helper.restore(self.model.parameters())
        return total_mse

    def reset(self):
        """ Resets the context of the model."""
        self.obs_context.clear()
        self.action_counter = self.action_seq_size

        self.bp_image_context.clear()
        self.inhand_image_context.clear()
        self.des_robot_pos_context.clear()

    @torch.no_grad()
    def predict(self, state: torch.Tensor, goal: Optional[torch.Tensor] = None, extra_args=None, if_vision=True) -> torch.Tensor:
        # scale data if necessarry, otherwise the scaler will return unchanged values

        if if_vision:
            bp_image, inhand_image, goal_image = state

            bp_image = torch.from_numpy(bp_image).to(self.device).float().permute(2, 0, 1).unsqueeze(0) / 255.
            inhand_image = torch.from_numpy(inhand_image).to(self.device).float().permute(2, 0, 1).unsqueeze(0) / 255.

            goal_image = goal_image.to(self.device).unsqueeze(0)

            # goal_image = torch.from_numpy(goal_image).to(self.device).float().permute(0, 3, 1, 2).unsqueeze(0) / 255.
            # des_robot_pos = torch.from_numpy(des_robot_pos).to(self.device).float().unsqueeze(0)

            # des_robot_pos = self.scaler.scale_input(des_robot_pos)

            self.bp_image_context.append(bp_image)
            self.inhand_image_context.append(inhand_image)
            # self.des_robot_pos_context.append(des_robot_pos)

            bp_image_seq = torch.stack(tuple(self.bp_image_context), dim=1)
            inhand_image_seq = torch.stack(tuple(self.inhand_image_context), dim=1)
            # des_robot_pos_seq = torch.stack(tuple(self.des_robot_pos_context), dim=1)

            input_state = (bp_image_seq, inhand_image_seq, goal_image)
        else:
            obs = torch.from_numpy(state).float().to(self.device).unsqueeze(0)
            obs = self.scaler.scale_input(obs)
            self.obs_context.append(obs)
            input_state = torch.stack(tuple(self.obs_context), dim=1)  # type: ignore

        if self.action_counter == self.action_seq_size:
            self.action_counter = 0

            if self.use_ema:
                self.ema_helper.store(self.model.parameters())
                self.ema_helper.copy_to(self.model.parameters())

            self.model.eval()

            model_pred = self.model(input_state, goal)

            # restore the previous model parameters
            if self.use_ema:
                self.ema_helper.restore(self.model.parameters())
            model_pred = self.scaler.inverse_scale_output(model_pred)

            self.curr_action_seq = model_pred

        next_action = self.curr_action_seq[:, self.action_counter, :]
        self.action_counter += 1
        return next_action.detach().cpu().numpy()

    def random_sample(self, state: torch.Tensor, goal: Optional[torch.Tensor] = None, extra_args=None, if_vision=True):
        # scale data if necessarry, otherwise the scaler will return unchanged values

        if if_vision:
            bp_imgs, inhand_imgs = state

            bp_imgs = bp_imgs[:, :self.obs_seq_len].contiguous()
            inhand_imgs = inhand_imgs[:, :self.obs_seq_len].contiguous()

            input_state = (bp_imgs, inhand_imgs)
        else:
            obs = torch.from_numpy(state).float().to(self.device).unsqueeze(0)
            obs = self.scaler.scale_input(obs)
            self.obs_context.append(obs)
            input_state = torch.stack(tuple(self.obs_context), dim=1)  # type: ignore

        # self.model.eval()
        # do default model evaluation
        model_pred, obs_embedding = self.model(input_state, goal, if_return_obs=True)

        return model_pred, obs_embedding
    @torch.no_grad()
    def load_pretrained_model(self, weights_path: str, sv_name=None, **kwargs) -> None:
        """
        Method to load a pretrained model weights inside self.model
        """
        # self.model.load_state_dict(torch.load(os.path.join(weights_path, "model_state_dict.pth")))
        self.model.load_state_dict(torch.load(os.path.join(weights_path, sv_name)), strict=False)
        self.ema_helper = ExponentialMovingAverage(self.model.parameters(), self.decay, self.device)
        log.info('Loaded pre-trained model parameters')

    @torch.no_grad()
    def store_model_weights(self, store_path: str, sv_name=None) -> None:
        """
        Store the model weights inside the store path as model_weights.pth
        """
        if self.use_ema:
            self.ema_helper.store(self.model.parameters())
            self.ema_helper.copy_to(self.model.parameters())
        # torch.save(self.model.state_dict(), os.path.join(store_path, "model_state_dict.pth"))
        torch.save(self.model.state_dict(), os.path.join(store_path, sv_name))
        if self.use_ema:
            self.ema_helper.restore(self.model.parameters())
        torch.save(self.model.state_dict(), os.path.join(store_path, "non_ema_model_state_dict.pth"))
