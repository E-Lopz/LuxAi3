from typing import Any, Dict

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
        self.observation_space = spaces.Box(-999, 999, shape=(11,))

    def observation(self, obs):
        self.state = self.env.state

        return SimpleUnitObservationWrapper.convert_obs(obs, self.env.state)

    # we make this method static so the submission/evaluation code can use this as well
    @staticmethod
    def convert_obs(obs: Dict[str, Any], env_cfg: Any) -> Dict[str, npt.NDArray]:
        observation = dict()

        for agent in obs.keys():
            obs_vec = np.zeros(11)  # Initial vector for each agent
            team_id = 0 if agent == "player_0" else 1

            shared_obs = obs[agent]
            relic_map = shared_obs["relic_nodes"]  # Access relic nodes directly
            relic_nodes_mask = shared_obs['relic_nodes_mask']
            
            # Find coordinates of elements not equal to [-1, -1] (visible relics)
            relic_tile_locations = []
            visible_relic_positions = relic_map[relic_nodes_mask == 1]
            relic_tile_locations.extend(visible_relic_positions)  # Append all visible relic positions

            # Convert the list of coordinates to a NumPy array
            relic_tile_locations = np.array(relic_tile_locations)
            map_features = shared_obs["map_features"]
            
            # Get the map size (width and height)
            energy_map = map_features["energy"]
            tile_type_map = map_features["tile_type"]
            # Get the map size (width, height)
            map_size = energy_map.shape  # or use another feature with the same shape
            width, height = energy_map.shape    

            units = shared_obs["units"]
            for i in range(16):
                # Extract unit's energy and position
                energy = units["energy"][team_id][i]
                pos = np.array(units["position"][team_id][i]) / [width, height]  # Normalize position

                # Get the current unit's tile energy and type
                x, y = int(pos[0] * width), int(pos[1] * height)

                current_energy = energy_map[x, y]
                current_tile_type = tile_type_map[x, y]

                # Get the energy and type for the top-left tile (0, 0)
                top_left_energy = energy_map[0, 0]
                top_left_tile_type = tile_type_map[0, 0]

                # Get the energy and type for the bottom-right tile (width-1, height-1)
                bottom_right_energy = energy_map[width-1, height-1]
                bottom_right_tile_type = tile_type_map[width-1, height-1]

                # Concatenate the features: current position, energy, and tile types
                tile_features = np.concatenate([
                    [current_energy, current_tile_type],
                    [top_left_energy, top_left_tile_type],
                    [bottom_right_energy, bottom_right_tile_type]
                ])

                # Normalize and concatenate the unit's position and energy with the tile features
                unit_vec = np.concatenate([pos, [energy], tile_features])

                # Compute closest relic tile
                if relic_tile_locations.size > 0:
                    # Calculate Euclidean distances from the unit to each relic tile
                    relic_tile_distances = np.sqrt(np.sum((relic_tile_locations - np.array(units["position"][team_id][i])) ** 2, axis=1))
                    
                    # Get the closest relic tile
                    closest_relic_tile_index = np.argmin(relic_tile_distances)
                    closest_relic_tile = relic_tile_locations[closest_relic_tile_index]
                    
                    # Normalize the coordinates of the closest relic tile
                    normalized_closest_relic_tile = closest_relic_tile / map_size
                else:
                    normalized_closest_relic_tile = np.zeros(2)  # If no relic tile is available

                # Concatenate the unit information and closest relic tile
                obs_vec = np.concatenate([unit_vec, normalized_closest_relic_tile - pos], axis=-1)
                
                break

            observation[agent] = obs_vec


        return observation