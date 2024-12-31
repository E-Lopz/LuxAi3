import sys
from typing import Any, Dict

import numpy as np
import numpy.typing as npt
from gymnasium import spaces
from gymnasium.spaces import MultiDiscrete
import jax
import jax.numpy as jnp


class Controller:
    def __init__(self, action_space: spaces.Space) -> None:
        self.action_space = MultiDiscrete([self.total_act_dims] * 16)  # 16 units, each with `total_act_dims` actions

    def action_to_lux_action(
        self, agent: str, obs: Dict[str, Any], action: npt.NDArray
    ):
        """
        Takes as input the current "raw observation" and the parameterized action and returns
        an action formatted for the Lux env
        """
        """
        Convert a multi-discrete action for 16 units into LuxAI environment actions.
        """
        shared_obs = obs[agent]
        lux_action = dict()
        units = shared_obs["units"][agent]

        for unit_id in range(16):
            if unit_id >= len(units):
                continue  # Skip if fewer than 16 units exist

            unit_action = action[unit_id]  # Get the action for this unit
            action_queue = []

            if self._is_move_action(unit_action):
                action_queue = [self._get_move_action(unit_action)]
            # else: no-op; don't update the action queue

            lux_action[unit_id] = action_queue  # Assign the unit's action queue

        return lux_action

    def action_masks(self, agent: str, obs: Dict[str, Any]):
        """
        Generate an action mask for all 16 units.
        """
        shared_obs = obs[agent]
        units = shared_obs["units"][agent]
        action_masks = np.zeros((16, self.total_act_dims), dtype=bool)

        for unit_id in range(16):
            if unit_id >= len(units):
                continue  # Skip if fewer than 16 units exist

            mask = np.zeros(self.total_act_dims, dtype=bool)
            mask[:self.move_act_dims] = True  # Movement actions are always valid
            mask[-1] = True  # No-op action is always valid
            action_masks[unit_id] = mask

        return action_masks


class SimpleUnitDiscreteController(Controller):
    def __init__(self, env) -> None:
        """
        A simple controller that controls only the robot that will get spawned.
        Moreover, it will always try to spawn one heavy robot if there are none regardless of action given

        For the robot unit
        - 4 cardinal direction movement (4 dims)
        - no op action (1 dim) - equivalent to not submitting an action queue which costs power

        It does not include
        -sapping

        To help understand how to this controller works to map one action space to the original lux action space,
        see how the lux action space is defined in luxai_s2/spaces/action.py

        """
        self.env = env
        self.move_act_dims = 4
        self.no_op_dims = 1

        self.move_dim_high = self.move_act_dims
        self.no_op_dim_high = self.move_dim_high + self.no_op_dims

        self.total_act_dims = self.no_op_dim_high
        action_space = MultiDiscrete([self.total_act_dims] * 16)

        super().__init__(action_space)

    def _is_move_action(self, id):
        return id < self.move_dim_high

    def _get_move_action(self, id):
        # move direction is id + 1 since we don't allow move center here
        return np.array([0, id + 1, 0, 0, 0, 1])

    def action_to_lux_action(
        self, agent: str, obs: Dict[str, Any], action: npt.NDArray
    ):
        shared_obs = obs[agent]
        lux_action = dict()
        #print(action, file=sys.stderr)

        # Map the focused unit's action for player_0, rest will be [0, 0, 0]
        action_mapping = {i: [i, 0, 0] for i in range(5)}  # Assuming you have 5 action choices
        
        # Initialize actions for all 16 units with [0, 0, 0] as the default action
        actions_player = [[0, 0, 0]] * 16  

        # Map the input array of 16 actions (player_0's actions) to the required format
        input_actions = action  # Assume this is an array of 16 actions
        for unit_id, unit_action in enumerate(input_actions):
            # Convert each unit's action using the `action_mapping`
            mapped_action = action_mapping[unit_action]  # Map the discrete action to [x, y, z]
            actions_player[unit_id] = mapped_action  # Update the action for the corresponding unit

        lux_action = actions_player
        # Convert the list of actions into a jax numpy array
        lux_action = jnp.array(lux_action)

        '''for unit_id in range(16):
            unit = units[unit_id]
            choice = action
            action_queue = []
            no_op = False
            if self._is_move_action(choice):
                action_queue = [self._get_move_action(choice)]
            else:
                # action is a no_op, so we don't update the action queue
                no_op = True

            # simple trick to help agents conserve power is to avoid updating the action queue
            # if the agent was previously trying to do that particular action already
            if len(unit["action_queue"]) > 0 and len(action_queue) > 0:
                same_actions = (unit["action_queue"][0] == action_queue[0]).all()
                if same_actions:
                    no_op = True
            if not no_op:
                lux_action[unit_id] = action_queue'''

            #break

        return lux_action

    def action_masks(self, agent: str, obs: Dict[str, Any]):
        """
        Defines a simplified action mask for this controller's action space

        Doesn't account for whether robot has enough power
        """
        team_id = 0 if agent == "player_0" else 1
        shared_obs = obs[agent]
        units = shared_obs["units"]
        action_masks = np.zeros((16, self.total_act_dims), dtype=bool)  # Action mask for all units

        map_features = shared_obs["map_features"]
            
        # Get the map size (width and height)
        energy_map = map_features["energy"]
        tile_type_map = map_features["tile_type"]
        map_size = energy_map.shape  # Get map size (width, height)
        width, height = energy_map.shape  

        # Iterate over each unit for the current team
        for i in range(16):
            action_mask = np.zeros(self.total_act_dims, dtype=bool)

            # Retrieve the unit's position
            x, y = units["position"][team_id][i]

            # Define movement actions: [0: Up, 1: Down, 2: Left, 3: Right]
            move_deltas = {
                1: (0, -1), # Up
                2: (1, 0),  # right
                3: (0, 1),  # down
                4: (-1, 0)  # left
            }

            # Check movement validity
            for move_action, delta in move_deltas.items():
                dx, dy = delta
                new_x, new_y = x + dx, y + dy

                # Ensure new position is within bounds and not an asteroid
                if 0 <= new_x < width and 0 <= new_y < height:
                    if tile_type_map[new_x, new_y] != 2:  # Assuming "asteroid" is a key in the map
                        action_mask[move_action] = True

            # No-op is always valid
            action_mask[0] = True

            # Assign the mask for this unit
            action_masks[i] = action_mask
            # Convert the action mask to numeric array if needed
            action_masks = action_masks.astype(np.float32)  # Ensures compatibility with JA
        return action_masks