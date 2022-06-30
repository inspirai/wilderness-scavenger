import random
from typing import Dict

import gym
import numpy as np
from gym import spaces
from ray.rllib.env import EnvContext
from inspirai_fps.utils import get_distance, get_position
from inspirai_fps.gamecore import Game
from inspirai_fps.gamecore import ActionVariable


class NavigationEnv(gym.Env):
    BASE_PORT = 50000
    ACT_VALS = {
        "walk_dir": [0, 90, 180, 270],
        "walk_speed": [0, 5, 10],
    }
    OBS_SPACE = spaces.Dict({
        "cur_loc": spaces.Box(low=-np.Inf, high=np.Inf, shape=(3,), dtype=np.float32),
        "tar_loc": spaces.Box(low=-np.Inf, high=np.Inf, shape=(3,), dtype=np.float32),
    })
    ACT_SPACE = spaces.Dict({
        k: spaces.Discrete(len(v)) for k, v in ACT_VALS.items()
    })
    
    def __init__(self, config: EnvContext):
        self.config = config
        self.render_scale = config.get("render_scale", 1)

        env_seed = config.get("random_seed", 0) + config.get("worker_index",0)
        self.seed(env_seed)

        self.observation_space = self.OBS_SPACE
        self.action_space = self.ACT_SPACE

        server_port = self.BASE_PORT + config.get("worker_index",0)
        print(f">>> New instance {self} on port: {server_port}")

        self.game = Game(
            map_dir=config["map_dir"],
            engine_dir=config["engine_dir"],
            server_port=server_port,
        )
        self.game.set_map_id(config["map_id"])
        self.game.set_episode_timeout(config["timeout"])
        self.game.set_random_seed(env_seed)
        self.game.turn_on_depth_map()
        self.game.set_game_mode(Game.MODE_NAVIGATION)
        self.game.set_available_actions([
            ActionVariable.WALK_DIR,
            ActionVariable.WALK_SPEED,
        ])
        self.start_location = None
        self.target_location = None

        locations = self.game.get_valid_locations()

        self.indoor_loc = locations["indoor"]
        self.outdoor_loc = locations["outdoor"]

        self.game.init()

    def _get_obs(self):
        return {
            "cur_loc": np.asarray(get_position(self.state)),
            "tar_loc": np.asarray(self.target_location)
        }

    def step(self, action_dict):
        action_vals = self._action_process(action_dict)
        self.game.make_action({0: action_vals})
        self.state = self.game.get_state()
        done = self.game.is_episode_finished()

        cur_loc = get_position(self.state)
        tar_loc = self.target_location

        if get_distance(cur_loc, tar_loc) <= self.game.target_trigger_distance:
            reward = 100
            done = True
        else:
            reward = 0

        if done:
            if self.print_log:
                Start = np.round(np.asarray(self.start_location), 2).tolist()
                Target = np.round(np.asarray(self.target_location), 2).tolist()
                End = np.round(np.asarray(get_position(self.state)), 2).tolist()
                Step = self.running_steps
                Reward = reward
                print(f"{Start=}\t{Target=}\t{End=}\t{Step=}\t{Reward=}")

        self.running_steps += 1

        return self._get_obs(), reward, done, {}

    def reset(self):
        print("Reset for a new game ...")
        self.start_location = random.choice(self.outdoor_loc)
        self.target_location = random.choice(self.outdoor_loc)
        self.game.set_start_location(self.start_location)
        self.game.set_target_location(self.target_location)
        self.game.new_episode()
        self.state = self.game.get_state()
        self.running_steps = 0
        return self._get_obs()

    def close(self):
        self.game.close()
        return super().close()

    def _action_process(self, action: Dict[str, int]):
        walk_dir = self.ACT_VALS["walk_dir"][action["walk_dir"]]
        walk_speed = self.ACT_VALS["walk_speed"][action["walk_speed"]]
        return [walk_dir, walk_speed]
