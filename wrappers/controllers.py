import sys
from typing import Any, Dict

import numpy as np
import numpy.typing as npt
from gymnasium import spaces
import jax
import jax.numpy as jnp


class Controller:
    def __init__(self, action_space: spaces.Space) -> None:
        self.action_space = action_space

    def action_to_lux_action(
        self, agent: str, obs: Dict[str, Any], action: npt.NDArray
    ):
        """
        Takes as input the current "raw observation" and the parameterized action and returns
        an action formatted for the Lux env
        """
        pass
        raise NotImplementedError()

    def action_masks(self, agent: str, obs: Dict[str, Any]):
        """
        Generates a boolean action mask indicating in each discrete dimension whether it would be valid or not
        """
        pass
        raise NotImplementedError()


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
        action_space = spaces.Discrete(self.total_act_dims)

        super().__init__(action_space)

    def _is_move_action(self, id):
        return id < self.move_dim_high

    def _get_move_action(self, id):
        # move direction is id + 1 since we don't allow move center here
        return np.array([0, id + 1, 0, 0, 0, 1])

    def action_to_lux_action(
        self, agent: str, obs: Dict[str, Any], action: npt.NDArray
    ):
        shared_obs = obs["player_0"]
        lux_action = dict()
        units = shared_obs["units"]
        print('action',action, file=sys.stderr)

        # Map the focused unit's action for player_0, rest will be [0, 0, 0]
        action_mapping = {i: [i, 0, 0] for i in range(5)}  # Assuming you have 5 action choices
        
        # Initialize actions for all 16 units with [0, 0, 0]
        actions_player_0 = [[0, 0, 0]] * 16  # Initialize 16 units with default [0, 0, 0]

        # Set the action for the first unit to [1, 0, 0] (or any desired action)
        focused_unit_action = action_mapping[action]  
        actions_player_0[0] = focused_unit_action  # Focus on the first unit of player_0
        lux_action = actions_player_0
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
        shared_obs = obs[agent]


        units = shared_obs["units"][agent]
        action_mask = np.zeros((self.total_act_dims), dtype=bool)
        for unit_id in units.keys():
            action_mask = np.zeros(self.total_act_dims)
            # movement is always valid
            #to change if out of bounds not valid
            action_mask[:4] = True

            # no-op is always valid
            action_mask[-1] = True
            break
        return action_mask