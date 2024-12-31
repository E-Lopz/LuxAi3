import os.path as osp

import gymnasium as gym
import numpy as np
import torch as th
import torch.nn as nn
from gymnasium import spaces
from gymnasium.wrappers import TimeLimit
from flax import struct
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import SubprocVecEnv, VecVideoRecorder
from stable_baselines3.ppo import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from wrappers import SimpleUnitDiscreteController, SimpleUnitObservationWrapper

from stable_baselines3.common.callbacks import (
    BaseCallback,
    CheckpointCallback,
    EvalCallback,
)

import luxai_s3.env
from luxai_s3.env import LuxAIS3Env
from luxai_s3.params import EnvParams
from luxai_s3.state import (
    ASTEROID_TILE,
    ENERGY_NODE_FNS,
    NEBULA_TILE,
    EnvObs,
    EnvState,
    MapTile,
    UnitState,
)
from luxai_s3.wrappers import LuxAIS3GymEnv

import jax
import jax.numpy as jnp

def get_opponent_model(model_path="/models/opponent_model.zip"):
    """
    Load an opponent model if the file exists.
    :param model_path: Path to the opponent model file.
    :return: Loaded model or None if the file doesn't exist.
    """
    if osp.exists(model_path):
        print(f"Loading opponent model from {model_path}")
        return PPO.load(model_path)
    else:
        print(f"No opponent model found at {model_path}. Defaulting to None.")
        return None


class CustomEnvWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, opponent_model=None, update_interval=10000, model_path='logs') -> None:
        """
        Adds a custom reward and turns the LuxAI_S3 environment into a single-agent environment for easy training.
        :param env: The base environment.
        :param opponent_model: The initial opponent model (can be None).
        :param update_interval: Number of steps after which the opponent model will be updated.
        :param model_path: Path to the directory where the opponent model is stored (best_model.zip).
        """
        super().__init__(env)
        self.prev_step_metrics = None
        self.opponent_model = opponent_model
        self.update_interval = update_interval
        self.model_path = model_path
        self.steps = 0  # Step counter to track updates

    def load_opponent_model(self):
        """
        Load the opponent model from the 'best_model.zip' file if it exists.
        """
        if self.model_path:
            opponent_model_path = osp.join(self.model_path, "models/best_model.zip")
            if osp.exists(opponent_model_path):
                print(f"Updating opponent model from {opponent_model_path}")
                self.opponent_model = PPO.load(opponent_model_path)
            else:
                print(f"No opponent model found at {opponent_model_path}. Skipping update.")
        else:
            print("Model path not provided. Skipping opponent model update.")

    def step(self, action):
        self.steps += 1
        # Update the opponent model if the update interval is reached
        if self.steps % self.update_interval == 0:
            self.load_opponent_model()

        state = self.env.state  # Ensure state is defined
        agent = "player_0"
        opp_agent = "player_1"

        action = {agent: action}
        action_mapping = {i: [i, 0, 0] for i in range(5)}  # Assuming you have 5 action choices

        # Use the opponent_model to determine opponent's actions
        if self.opponent_model:
            # Predict the opponent's actions for all 16 units
            opponent_actions, _ = self.opponent_model.predict(self.opp_obs, deterministic=True)

            # Initialize actions for all 16 units with [0, 0, 0] as the default action
            actions_player_1 = [[0, 0, 0]] * 16

            # Map the predicted actions to the required format
            for unit_id, unit_action in enumerate(opponent_actions):
                # Convert each unit's predicted action using the `action_mapping`
                mapped_action = action_mapping[int(unit_action)]  # Ensure the action is cast to an int
                actions_player_1[unit_id] = mapped_action  # Update the action for the corresponding unit

            # Convert the updated list of actions into a jax numpy array
            action["player_1"] = jnp.array(actions_player_1)
        else:
            action["player_1"] = jnp.array([[0, 0, 0]] * 16)  # 16 units with default [0, 0, 0] action for player_1

        # Initialize actions for all 16 units with [0, 0, 0] as the default action
        actions_player_0 = [[0, 0, 0]] * 16  

        # Map the input array of 16 actions (player_0's actions) to the required format
        input_actions = action["player_0"]  # Assume this is an array of 16 actions
        for unit_id, unit_action in enumerate(input_actions):
            # Convert each unit's action using the `action_mapping`
            mapped_action = action_mapping[unit_action]  # Map the discrete action to [x, y, z]
            actions_player_0[unit_id] = mapped_action  # Update the action for the corresponding unit

        # Convert the updated list of actions into a jax numpy array
        action["player_0"] = jnp.array(actions_player_0)

        obs, _, termination, truncation, info = self.env.step(action)
        self.opp_obs = obs['player_1']

        done = {k: termination[k] or truncation[k] for k in termination}
        obs = obs[agent]

       

        metrics = {
            "points_produced": state.team_points[0],
            'enemy_points': state.team_points[1],
            "energy": sum(state.units.energy[0])[0]/1000,
            "enemy_energy": sum(state.units.energy[1])[0]/1000,
            "win": state.team_wins[0],
            'lost': state.team_wins[1],
        }
        energy_end = 0
        advantage = 0
        if state.match_steps == 100:
            energy_end = 1 if metrics["energy"] > metrics["enemy_energy"] else -1
            advantage = 1 if metrics["points_produced"] > metrics["enemy_points"] else -1

        info["metrics"] = metrics
        reward = 0
        if self.prev_step_metrics is not None:
            roundWin , matchWin = 0 , 0 
            if metrics["win"] > self.prev_step_metrics["win"]:
                roundWin = 1 
                if metrics["win"] >= 3:
                    matchWin = 1
             
            if metrics["lost"] > self.prev_step_metrics["lost"]:
                roundWin = -1 
                if metrics["win"] >= 3:
                    matchWin = -1

            # Calculate advantage rewards
            point_advantage = metrics["points_produced"] - metrics["enemy_points"]
            energy_advantage = metrics["energy"] - metrics["enemy_energy"]

            # Non-terminal advantage signals
            advantage_reward = 0
            if point_advantage > 0:
                advantage_reward += 0.01 * point_advantage  # Positive reward for point advantage
            elif point_advantage < 0:
                advantage_reward -= 0.01 * abs(point_advantage)  # Penalize for losing point advantage

            if energy_advantage > 0:
                advantage_reward += 0.005 * energy_advantage  # Reward for energy advantage
            elif energy_advantage < 0:
                advantage_reward -= 0.005 * abs(energy_advantage)  # Penalize for energy disadvantage
            reward = (
                0.006 * (metrics["points_produced"] - self.prev_step_metrics["points_produced"])
                + 0.002 * (metrics["energy"] - self.prev_step_metrics["energy"])
                + advantage_reward
                + 0.05 * advantage  # End-of-match advantage
                + 0.05 * energy_end
                + 0.2 * roundWin
                + 1.0 * matchWin
            )
        self.prev_step_metrics = metrics.copy()
        return obs, reward, termination[agent], truncation[agent], info

    def reset(self, **kwargs):
        # Reset the wrapped environment
        obs, reset_info = self.env.reset(**kwargs)

        # Initialize self.opp_obs with the initial observation for player_1
        self.opp_obs = obs['player_1']

        # Return the observation for the active agent (player_0)
        return obs["player_0"], reset_info
    
def parse_args():
    import argparse

    parser = argparse.ArgumentParser(
        description="Simple script that simplifies Lux AI Season 4 as a single-agent environment with a reduced observation and action space. It trains a policy that can succesfully control a unit to collect points and stay alive"
    )
    parser.add_argument("-s", "--seed", type=int, default=12, help="seed for training")
    parser.add_argument(
        "-n",
        "--n-envs",
        type=int,
        default=8,
        help="Number of parallel envs to run. Note that the rollout size is configured separately and invariant to this value",
    )
    parser.add_argument(
        "--max-episode-steps",
        type=int,
        default=500,
        help="Max steps per episode before truncating them",
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=5_000_000,
        help="Total timesteps for training",
    )

    parser.add_argument(
        "--eval",
        action="store_true",
        help="If set, will only evaluate a given policy. Otherwise enters training mode",
    )
    parser.add_argument(
        "--model-path", type=str, help="Path to SB3 model weights to use for evaluation"
    )
    parser.add_argument(
        "-l",
        "--log-path",
        type=str,
        default="logs",
        help="Logging path",
    )
    args = parser.parse_args()
    return args

def make_env(env_id: str, rank: int, seed: int = 0, max_episode_steps=100, opponent_model = None):
    def _init() -> gym.Env:
        # verbose = 0
        # collect stats so we can create reward functions
        #env = gym.make(env_id, verbose=0, collect_stats=True, disable_env_checker=True)
        env = LuxAIS3Env(auto_reset=False)
        env = LuxAIS3GymEnv(
            env
        )
        controller = SimpleUnitDiscreteController(env)
        action_space = controller.action_space
        env.action_space = action_space
        env = SimpleUnitObservationWrapper(
            env
        )  # changes observation to include a few simple features
        env = CustomEnvWrapper(env,opponent_model=opponent_model)  # convert to single agent, add our reward
        env = TimeLimit(
            env, max_episode_steps=max_episode_steps
        )  # set horizon to 100 to make training faster. Default is 1000
        env = Monitor(env)  # for SB3 to allow it to record metrics
        env.reset(seed=seed + rank)
        set_random_seed(seed)
        return env

    return _init

class TensorboardCallback(BaseCallback):
    def __init__(self, tag: str, verbose=0):
        super().__init__(verbose)
        self.tag = tag

    def _on_step(self) -> bool:
        c = 0

        for i, done in enumerate(self.locals["dones"]):
            if done:
                info = self.locals["infos"][i]
                c += 1
                for k in info["metrics"]:
                    stat = info["metrics"][k]
                    self.logger.record_mean(f"{self.tag}/{k}", stat)
        return True

def save_model_state_dict(save_path, model):
    # save the policy state dict for kaggle competition submission
    state_dict = model.policy.to("cpu").state_dict()
    th.save(state_dict, save_path)

def evaluate(args, env_id, model):
    model = model.load(args.model_path)
    video_length = 1000  # default horizon
    eval_env = SubprocVecEnv(
        [make_env(env_id, i, max_episode_steps=1000) for i in range(args.n_envs)]
    )
    eval_env = VecVideoRecorder(
        eval_env,
        osp.join(args.log_path, "eval_videos"),
        record_video_trigger=lambda x: x == 0,
        video_length=video_length,
        name_prefix=f"evaluation_video",
    )
    eval_env.reset()
    out = evaluate_policy(model, eval_env, render=False, deterministic=False)
    print(out)

def train(args, env_id, model: PPO, opponent_model = None):
    eval_env = SubprocVecEnv(
        [make_env(env_id, i, max_episode_steps=1000, opponent_model=opponent_model) for i in range(4)]
    )
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=osp.join(args.log_path, "models"),
        log_path=osp.join(args.log_path, "eval_logs"),
        eval_freq=24_000,
        deterministic=False,
        render=False,
        n_eval_episodes=5,
    )

    model.learn(
        args.total_timesteps,
        callback=[TensorboardCallback(tag="train_metrics"), eval_callback],
    )
    model.save(osp.join(args.log_path, "models/latest_model"))



def main(args):
    print("Training with args", args)
    if args.seed is not None:
        set_random_seed(args.seed)

    # Load opponent model
    # Define the opponent model path dynamically
    opponent_model_path = osp.join(args.log_path, "models/best_model.zip")
    opponent_model = get_opponent_model(opponent_model_path)
    env_id = "LuxAI_S3"
    env = SubprocVecEnv(
        [
            make_env(env_id, i, max_episode_steps=args.max_episode_steps, opponent_model=opponent_model)
            for i in range(args.n_envs)
        ]
    )
    env.reset()

    rollout_steps = 10000
    policy_kwargs = dict(net_arch=(128, 128))
    model = PPO(
        "MlpPolicy",
        env,
        n_steps=rollout_steps // args.n_envs,
        batch_size=1000,
        learning_rate=3e-4,
        policy_kwargs=policy_kwargs,
        verbose=1,
        n_epochs=2,
        target_kl=0.05,
        gamma=0.995,
        tensorboard_log=osp.join(args.log_path),
    )
    if args.eval:
        evaluate(args, env_id, model)
    else:
        train(args, env_id, model, opponent_model)

if __name__ == "__main__":
    # python ../examples/sb3.py -l logs/exp_1 -s 42 -n 1
    main(parse_args())