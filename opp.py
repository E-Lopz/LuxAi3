from lux.utils import direction_to
import sys
import numpy as np

"""
This file is where your agent's logic is kept. Define a bidding policy, factory placement policy, as well as a policy for playing the normal phase of the game

The tutorial will learn an RL agent to play the normal phase and use heuristics for the other two phases.

Note that like the other kits, you can only debug print to standard error e.g. print("message", file=sys.stderr)
"""

import os.path as osp
import sys
import numpy as np
import torch as th
from stable_baselines3 import PPO
from lux.kit import from_json
from wrappers import SimpleUnitDiscreteController, SimpleUnitObservationWrapper

# change this to use weights stored elsewhere
# make sure the model weights are submitted with the other code files
# any files in the logs folder are not necessary. Make sure to exclude the .zip extension here
MODEL_WEIGHTS_RELATIVE_PATH = "./opp_model"

class Agent:
    def __init__(self, player: str, env_cfg) -> None:
        self.player = player
        self.opp_player = "player_1" if self.player == "player_0" else "player_0"
        self.team_id = 0 if self.player == "player_0" else 1
        self.opp_team_id = 1 if self.team_id == 0 else 0
        np.random.seed(0)
        self.env_cfg = env_cfg

        directory = osp.dirname(__file__)
        self.policy = PPO.load(osp.join(directory, MODEL_WEIGHTS_RELATIVE_PATH))

        self.controller = SimpleUnitDiscreteController(self.env_cfg)

    def act(self, step: int, obs, remainingOverageTime: int = 60):
        # first convert observations using the same observation wrapper you used for training
        # note that SimpleUnitObservationWrapper takes input as the full observation for both players and returns an obs for players
        raw_obs = dict(player_0=obs, player_1=obs)
        obs = SimpleUnitObservationWrapper.convert_obs(raw_obs, env_cfg=self.env_cfg)
        obs = obs[self.player]

        

        obs = th.from_numpy(obs).float()
        with th.no_grad():

            # to improve performance, we have a rule based action mask generator for the controller used
            # which will force the agent to generate actions that are valid only.
            #action_mask = (
            #    th.from_numpy(self.controller.action_masks(self.team_id, raw_obs))
            #    .unsqueeze(0)
            #    .bool()
            #)
            
            # SB3 doesn't support invalid action masking. So we do it ourselves here
            # Extract features for the current observation
            features = self.policy.policy.features_extractor(obs.unsqueeze(0))

            # Step 2: Check latent features
            latent_pi, _ = self.policy.policy.mlp_extractor(features)


            # Step 3: Check logits
            logits = self.policy.policy.action_net(latent_pi)

            # Reshape logits to [16, total_act_dims]
            logits_per_unit = logits.view(16, 5)


            # Step 4: Check action mask
            action_mask = (
                th.from_numpy(self.controller.action_masks(self.player, raw_obs))
                .bool()
            )



            # Step 5: Check masked logits
            logits_per_unit[~action_mask] = -1e8
            

            # Step 6: Check sampled actions
            dist = th.distributions.Categorical(logits=logits_per_unit)
            actions = dist.sample().cpu().numpy()

            

        # use our controller which we trained with in train.py to generate a Lux S3 compatible action
        lux_action = self.controller.action_to_lux_action(
            self.player, raw_obs, actions.tolist()
        )

        return lux_action