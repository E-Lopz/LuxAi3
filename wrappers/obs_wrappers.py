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

def update_map_features(width,height,tile_type_map,energy_map,memory_map):
    # Check if new tiles are discovered and update memory        
    # Example logic for updating map memory based on visibility
    for x in range(width):
        for y in range(height):
            if tile_type_map[x][y] != -1:  # If the tile is discovered
                # Update map features
                memory_map['energy'][x][y] = energy_map[x][y]
                memory_map["tile_type"][x][y] = tile_type_map[x][y]
    return memory_map


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
        self.observation_space = spaces.Box(-999, 999, shape=(16,1167))
        self.memory = {
            'player_0': {
                "relics": set(),
                'map': {},
                'last_turn_points': 0,  # Store points gained in the last turn
                'previous_points': 0   # Track points from the previous step for difference calculation
            },
            'player_1': {
                "relics": set(),
                'map': {},
                'last_turn_points': 0,  # Same as for player_0
                'previous_points': 0   # Same as for player_0
            },
            'accessed': 0
        }
        

    def observation(self, obs):
        self.state = self.env.state

        return SimpleUnitObservationWrapper.convert_obs(obs, self.env.state, self.memory)
    
    def eraseMemory(self):
        self.memory = {
            'player_0': {
                "relics": set(),
                'map': {},
                'last_turn_points': 0,  # Store points gained in the last turn
                'previous_points': 0   # Track points from the previous step for difference calculation
            },
            'player_1': {
                "relics": set(),
                'map': {},
                'last_turn_points': 0,  # Same as for player_0
                'previous_points': 0   # Same as for player_0
            },
            'accessed': 0
        }
        

                    




    # Modify the method to include game environment information
    @staticmethod
    def convert_obs(obs: Dict[str, Any], env_cfg: Any, memory: Dict) -> Dict[str, npt.NDArray]:
        memory['accessed']+=1
        observation = dict()
        for agent in obs.keys():
            observation[agent] = np.zeros((16, 1167))  # Update shape to accommodate extra game info (e.g., 2 additional features)
            team_id = 0 if agent == "player_0" else 1
            opp_team_id = 1 if team_id == 0 else 0
            shared_obs = obs[agent]

            # Get the map size (width and height)
            map_features = shared_obs["map_features"]
            energy_map = map_features["energy"]
            tile_type_map = map_features["tile_type"]
            map_size = energy_map.shape  # Get map size (width, height)
            width, height = energy_map.shape

            relic_map = shared_obs["relic_nodes"]  # Access relic nodes directly
            relic_nodes_mask = shared_obs['relic_nodes_mask']
            total_relics = len(relic_nodes_mask)

            if not memory[agent]['map']:
                memory[agent]['map'] = {
                    "energy": np.full((width, height), -1, dtype=int),  # Initialize energy grid with -1
                    "tile_type": np.full((width, height), -1, dtype=int)  # Initialize tile type grid with -1
                    }
                memory[agent]["relics"] = np.full((total_relics, 2), -1, dtype=float)
                
                         
            memory_map = memory[agent]['map']
            memory_map=update_map_features(width,height,tile_type_map,energy_map,memory_map)
            memory[agent]['map']=memory_map
            # Concatenate updated map data with the rest of the features
            energy_map_flat = memory_map['energy'].flatten()  # Flatten 24x24 energy map to a 1D array
            tile_type_map_flat = memory_map['tile_type'].flatten()

            # Update visible relic positions
            visible_relic_positions = relic_map[relic_nodes_mask == 1]
            for i, relic_pos in enumerate(visible_relic_positions):
                if i < memory[agent]["relics"].shape[0]:
                    memory[agent]["relics"][i] = relic_pos 




            # Extract global game information
            team_points = shared_obs["team_points"][team_id]
            team_wins = shared_obs["team_wins"][team_id]
            match_steps = shared_obs["match_steps"]
            enemy_points = shared_obs["team_points"][opp_team_id]
            enemy_wins = shared_obs["team_wins"][opp_team_id]
            team_wins = shared_obs["team_wins"][team_id]
            match_steps = shared_obs["match_steps"]

            # Convert memory["relics"] to a NumPy array
            relic_tile_locations = np.array(memory[agent]["relics"])
            visible_relics = len(relic_tile_locations)

            units = shared_obs["units"]
            team_energy = sum(units["energy"][team_id])
            enemy_energy = sum(units["energy"][opp_team_id])



            if match_steps % 100 == 0:  # Reset at the start of a new round
                memory[agent]["last_turn_points"] = 0
            else:
                memory[agent]["last_turn_points"] = team_points - memory[agent].get("previous_points", 0)
                memory[agent]["previous_points"] = team_points

            last_turn_points = memory[agent]["last_turn_points"]


            for i in range(16):
                # Extract unit's energy and position
                energy = units["energy"][team_id][i]
                pos = np.array(units["position"][team_id][i])

                if relic_tile_locations.size > 0:
                    # Calculate distances to relics stored in memory
                    relic_tile_distances = np.sqrt(
                        np.sum((relic_tile_locations - pos) ** 2, axis=1)
                    )
                    closest_relic_tile = relic_tile_locations[np.argmin(relic_tile_distances)]
                    closest_relic_tile = closest_relic_tile
                else:
                    closest_relic_tile = np.full(2, -1)
                # Combine unit features and game-level information
                unit_vec = np.concatenate([pos, [energy]])
                obs_vec = np.concatenate([
                    unit_vec,
                    closest_relic_tile - pos,  # Relic tile information
                    [team_points ,last_turn_points , match_steps / 100, team_wins/5,team_energy,team_id,visible_relics/total_relics,
                     enemy_points, enemy_wins/5, enemy_energy],tile_type_map_flat,energy_map_flat  # Add normalized team points and match step
                ])
                observation[agent][i] = obs_vec

        return observation
    