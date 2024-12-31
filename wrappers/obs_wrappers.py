from typing import Any, Dict
import sys


import gymnasium as gym
import numpy as np
import numpy.typing as npt
from gymnasium import spaces

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

def append_unique_relics(existing_positions, new_positions):
    for position in new_positions:
        # Check if the position is already in the list
        if not any(np.array_equal(position, existing) for existing in existing_positions):
            existing_positions.append(position)


class SimpleUnitObservationWrapper(gym.ObservationWrapper):
    """
    A simple state based observation to work with in pair with the SimpleUnitDiscreteController

    It contains info only on the first robot If there are no owned robots the observation is just zero.
    No information about the opponent is included. This will generate observations for all teams.

    Included features:
    - First robot's stats
    - distance vector to closest relic


    """ 

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self.state = None
        self.observation_space = spaces.Box(-999, 999, shape=(16,24))
        memory = {"relics": set()}

    def observation(self, obs):
        self.state = self.env.state

        return SimpleUnitObservationWrapper.convert_obs(obs, self.env.state)

    # Modify the method to include game environment information
    @staticmethod
    def convert_obs(obs: Dict[str, Any], env_cfg: Any) -> Dict[str, npt.NDArray]:
        observation = dict()
        for agent in obs.keys():
            observation[agent] = np.zeros((16, 24))  # Update shape to accommodate extra game info (e.g., 2 additional features)
            team_id = 0 if agent == "player_0" else 1
            opp_team_id = 1 if team_id == 0 else 0

            shared_obs = obs[agent]
            relic_map = shared_obs["relic_nodes"]  # Access relic nodes directly
            relic_nodes_mask = shared_obs['relic_nodes_mask']

            # Extract global game information
            team_points = shared_obs["team_points"][team_id]
            team_wins = shared_obs["team_wins"][team_id]
            match_steps = shared_obs["match_steps"]
            enemy_points = shared_obs["team_points"][opp_team_id]
            enemy_wins = shared_obs["team_wins"][opp_team_id]
            team_wins = shared_obs["team_wins"][team_id]
            match_steps = shared_obs["match_steps"]

            
            # Find coordinates of visible relics
            relic_tile_locations = []
            visible_relic_positions = relic_map[relic_nodes_mask == 1]
            relic_tile_locations.extend(visible_relic_positions)  # Append all visible relic positions

            # Convert the list of coordinates to a NumPy array
            relic_tile_locations = np.array(relic_tile_locations)
            relic_tile_locations
            visible_relics = len(relic_tile_locations)
            map_features = shared_obs["map_features"]
            
            # Get the map size (width and height)
            energy_map = map_features["energy"]
            tile_type_map = map_features["tile_type"]
            map_size = energy_map.shape  # Get map size (width, height)
            width, height = energy_map.shape    

            units = shared_obs["units"]
            team_energy = sum(units["energy"][team_id])
            enemy_energy = sum(units["energy"][opp_team_id])
            for i in range(16):
                # Extract unit's energy and position
                energy = units["energy"][team_id][i]
                pos = np.array(units["position"][team_id][i]) / [width, height]  # Normalize position

                # Get the current unit's tile energy and type
                x, y = int(pos[0] * width), int(pos[1] * height)
                current_energy = energy_map[x, y]
                current_tile_type = tile_type_map[x, y]

                # Initialize variables with default values
                top_energy = -1
                top_tile_type = -1
                bottom_energy = -1
                bottom_tile_type = -1
                left_energy = -1
                left_tile_type = -1
                right_energy = -1
                right_tile_type = -1

                # Check bounds and update variables
                # Assuming `energy_map` and `tile_type_map` have dimensions (width, height)
                width, height = energy_map.shape

                # Top
                if y - 1 >= 0:
                    top_energy = energy_map[x, y - 1]
                    top_tile_type = tile_type_map[x, y - 1]

                # Bottom
                if y + 1 < height:
                    bottom_energy = energy_map[x, y + 1]
                    bottom_tile_type = tile_type_map[x, y + 1]

                # Left
                if x - 1 >= 0:
                    left_energy = energy_map[x - 1, y]
                    left_tile_type = tile_type_map[x - 1, y]

                # Right
                if x + 1 < width:
                    right_energy = energy_map[x + 1, y]
                    right_tile_type = tile_type_map[x + 1, y]

                # Concatenate the features: current position, energy, and tile types
                tile_features = np.concatenate([
                    [current_energy, current_tile_type],
                    [top_energy, top_tile_type],
                    [bottom_energy, bottom_tile_type],
                    [left_energy, left_tile_type],
                    [right_energy, right_tile_type],
                ])

                if relic_tile_locations.size > 0:
                    relic_tile_distances = np.sqrt(
                        np.sum((relic_tile_locations - np.array(units["position"][team_id][i])) ** 2, axis=1)
                    )
                    closest_relic_tile = relic_tile_locations[np.argmin(relic_tile_distances)]
                    normalized_closest_relic_tile = closest_relic_tile / map_size
                else:
                    normalized_closest_relic_tile = np.zeros(2)

                # Combine unit features and game-level information
                unit_vec = np.concatenate([pos, [energy], tile_features])
                obs_vec = np.concatenate([
                    unit_vec,
                    normalized_closest_relic_tile - pos,  # Relic tile information
                    [team_points, match_steps / 100, team_wins/5,team_energy,team_id,visible_relics,
                     enemy_points, enemy_wins/5, enemy_energy]  # Add normalized team points and match step
                ])
                observation[agent][i] = obs_vec

        return observation