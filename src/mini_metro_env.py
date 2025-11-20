import gymnasium as gym
from gymnasium import spaces
import numpy as np
import itertools
from typing import Dict, Any

from mediator import Mediator
from config import num_paths, screen_width, screen_height, screen_color
from geometry.type import ShapeType

import pygame

MAX_STATIONS = 20
MAX_PATHS = num_paths
MAX_STATIONS_PER_PATH = 12

class MetroGameEnv(gym.Env):
    """A Gymnasium environment for the Python Mini Metro game."""

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, render_mode: str | None = None):
        super().__init__()
        self.mediator = Mediator()
        
        self.render_mode = render_mode
        self.screen = None
        self.clock = None
        if self.render_mode == "human":
            pygame.init()
            pygame.display.set_caption("Metro RL Training")
            self.screen = pygame.display.set_mode((screen_width, screen_height))
            self.clock = pygame.time.Clock()

        self.shape_types = sorted([e.value for e in ShapeType])
        self.shape_to_idx = {shape: i for i, shape in enumerate(self.shape_types)}
        self.num_shape_types = len(self.shape_types)
        
        self._action_map = self._create_action_map()
        self._action_reverse_map = {}
        for action_id, info in self._action_map.items():
            if info["type"] == "CREATE_OR_EXTEND_PATH":
                key = ("CREATE", info["start_idx"], info["end_idx"])
                self._action_reverse_map[key] = action_id
            elif info["type"] == "INSERT_STATION":
                key = ("INSERT", info["insert_idx"], info["exist1_idx"], info["exist2_idx"])
                self._action_reverse_map[key] = action_id
        self.action_space = spaces.Discrete(len(self._action_map))

        self.observation_space = self._create_observation_space()

    def _get_action_mask(self) -> np.ndarray:
        mask = np.zeros(self.action_space.n, dtype=np.int8)
        mask[0] = 1

        stations = self.mediator.stations
        num_stations = len(stations)
        
        for i in range(num_stations):
            for j in range(num_stations):
                if i == j: continue
                action_id = self._action_reverse_map.get(("CREATE", i, j))
                if action_id is not None:
                    mask[action_id] = 1

        for path in self.mediator.paths:
            if len(path.stations) < 2: continue
            
            edges = []
            for k in range(len(path.stations) - 1):
                edges.append((path.stations[k], path.stations[k+1]))
            if path.is_looped:
                edges.append((path.stations[-1], path.stations[0]))

            for s1, s2 in edges:
                s1_idx = stations.index(s1)
                s2_idx = stations.index(s2)
                
                for i in range(num_stations):
                    s_insert = stations[i]
                    if s_insert not in path.stations:
                        action_id = self._action_reverse_map.get(("INSERT", i, s1_idx, s2_idx))
                        if action_id is not None:
                            mask[action_id] = 1
                            
        return mask

    def render(self):
        if self.render_mode != "human" or self.screen is None: 
            return
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return

        self.screen.fill(screen_color)
        self.mediator.render(self.screen)
        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
            self.screen = None
            self.clock = None

    def _create_action_map(self) -> Dict[int, Dict[str, Any]]:
        """Creates the mapping from discrete action int to game action."""
        action_map = {0: {"type": "NO_OP"}}
        action_id = 1
        
        station_pairs = list(itertools.permutations(range(MAX_STATIONS), 2))
        for start_idx, end_idx in station_pairs:
            action_map[action_id] = {"type": "CREATE_OR_EXTEND_PATH", "start_idx": start_idx, "end_idx": end_idx}
            action_id += 1
            
        station_trios = list(itertools.permutations(range(MAX_STATIONS), 3))
        for insert_idx, exist1_idx, exist2_idx in station_trios:
            action_map[action_id] = {"type": "INSERT_STATION", "insert_idx": insert_idx, "exist1_idx": exist1_idx, "exist2_idx": exist2_idx}
            action_id += 1
            
        return action_map

    def _create_observation_space(self) -> spaces.Dict:
        """
        Creates a Dict space suitable for GNNs.
        """
        self.features_per_node = 6 + (2 * self.num_shape_types)
        
        self.max_edges = (MAX_PATHS * MAX_STATIONS_PER_PATH) * 2

        return spaces.Dict({
            "node_features": spaces.Box(
                low=-1.0, 
                high=np.inf, 
                shape=(MAX_STATIONS, self.features_per_node), 
                dtype=np.float32
            ),
            
            "edge_index": spaces.Box(
                low=-1, 
                high=MAX_STATIONS, 
                shape=(2, self.max_edges), 
                dtype=np.int64
            ),
            
            "node_mask": spaces.Box(
                low=0, 
                high=1, 
                shape=(MAX_STATIONS,), 
                dtype=np.int8
            )
        })

    def _get_obs(self) -> Dict[str, np.ndarray]:
        node_feats = np.zeros((MAX_STATIONS, self.features_per_node), dtype=np.float32)
        node_mask = np.zeros((MAX_STATIONS,), dtype=np.int8)
        
        stations_in_paths = set()
        for path in self.mediator.paths:
            for station in path.stations:
                stations_in_paths.add(station.id)

        station_id_to_idx = {}

        for i in range(MAX_STATIONS):
            if i < len(self.mediator.stations):
                station = self.mediator.stations[i]
                station_id_to_idx[station.id] = i
                node_mask[i] = 1
                
                feat_idx = 0
                
                # Feature 1: Exists
                node_feats[i, feat_idx] = 1.0
                feat_idx += 1
                
                # Feature 2: Is Connected
                node_feats[i, feat_idx] = 1.0 if station.id in stations_in_paths else 0.0
                feat_idx += 1
                
                # Feature 3 & 4: Position (Normalized)
                node_feats[i, feat_idx] = station.position.left / screen_width
                feat_idx += 1
                node_feats[i, feat_idx] = station.position.top / screen_height
                feat_idx += 1
                
                # Feature 5: Overcrowded
                node_feats[i, feat_idx] = 1.0 if station.is_overcrowded else 0.0
                feat_idx += 1
                
                # Feature 6: Overcrowd Timer
                if station.is_overcrowded:
                    elapsed = self.mediator.time_ms - station.overcrowd_start_time_ms
                    node_feats[i, feat_idx] = min(elapsed / 10000.0, 1.0)
                feat_idx += 1
                
                # Feature 7: Station Type (One-Hot)
                shape_idx = self.shape_to_idx[station.shape.type.value]
                node_feats[i, feat_idx + shape_idx] = 1.0
                feat_idx += self.num_shape_types
                
                # Feature 8: Passengers (Count per destination type)
                for p in station.passengers:
                    dest_idx = self.shape_to_idx[p.destination_shape.type.value]
                    node_feats[i, feat_idx + dest_idx] += (1.0 / max(station.capacity, 1))

        edge_list = []
        
        for path in self.mediator.paths:
            num_stations = len(path.stations)
            if num_stations < 2:
                continue
                
            for k in range(num_stations - 1):
                s1 = path.stations[k]
                s2 = path.stations[k+1]
                
                u = station_id_to_idx[s1.id]
                v = station_id_to_idx[s2.id]
                
                edge_list.append([u, v])
                edge_list.append([v, u])
            
            if path.is_looped:
                s_last = path.stations[-1]
                s_first = path.stations[0]
                
                u = station_id_to_idx[s_last.id]
                v = station_id_to_idx[s_first.id]
                
                edge_list.append([u, v])
                edge_list.append([v, u])

        edge_index = np.full((2, self.max_edges), -1, dtype=np.int64)
        
        if len(edge_list) > 0:
            edges_arr = np.array(edge_list, dtype=np.int64)
            
            edges_arr = edges_arr.T 
            
            num_actual_edges = min(edges_arr.shape[1], self.max_edges)
            edge_index[:, :num_actual_edges] = edges_arr[:, :num_actual_edges]

        return {
            "node_features": node_feats,
            "edge_index": edge_index,
            "node_mask": node_mask
        }

    def _get_info(self) -> Dict[str, Any]:
        """Returns info dict, including the crucial action mask."""
        return {
            "score": self.mediator.score,
            "steps": self.mediator.steps,
            "action_mask": self._get_action_mask()
        }

    def reset(self, seed=None, options=None) -> tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        self.mediator = Mediator()
        if self.render_mode == "human":
            self.render()
        return self._get_obs(), self._get_info()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        prev_score = self.mediator.score
        num_loops_before = sum(1 for p in self.mediator.paths if p.is_looped)
        
        action_info = self._action_map.get(action)
        action_was_valid = False
        
        if action_info:
            action_type = action_info["type"]
            if action_type == "NO_OP":
                action_was_valid = True

            elif action_type == "CREATE_OR_EXTEND_PATH":
                start_idx, end_idx = action_info["start_idx"], action_info["end_idx"]
                if start_idx < len(self.mediator.stations) and end_idx < len(self.mediator.stations) and start_idx != end_idx:
                    start_station = self.mediator.stations[start_idx]
                    end_station = self.mediator.stations[end_idx]
                    action_was_valid = self.mediator.create_or_extend_path(start_station, end_station)

            elif action_type == "INSERT_STATION":
                insert_idx, exist1_idx, exist2_idx = action_info["insert_idx"], action_info["exist1_idx"], action_info["exist2_idx"]
                if all(i < len(self.mediator.stations) for i in [insert_idx, exist1_idx, exist2_idx]):
                    s_insert = self.mediator.stations[insert_idx]
                    s1 = self.mediator.stations[exist1_idx]
                    s2 = self.mediator.stations[exist2_idx]
                    action_was_valid = self.mediator.insert_station_on_path(s_insert, s1, s2)

        # Simulate 15 game-ticks
        for _ in range(15): 
            if self.mediator.is_game_over: break
            self.mediator.increment_time(16)

        reward = (self.mediator.score - prev_score) * 25.0 

        if not action_was_valid:
            reward -= 0.5

        terminated = self.mediator.is_game_over

        if terminated:
            reward -= 50.0
        
        if self.render_mode == "human":
            self.render()

        return self._get_obs(), reward, terminated, False, self._get_info()
    