import gym
from gym import spaces
from inspirai_fps import Game, ActionVariable
from inspirai_fps.utils import get_distance, get_position
from ray.rllib.env import EnvContext

import numpy as np
import cv2

BASE_PORT = 80000


def create_game(env_config):
    map_dir = env_config["map_dir"]
    engine_dir = env_config["engine_dir"]
    num_envs_per_worker = env_config.get("num_envs_per_worker", 1)
    port = (
        BASE_PORT
        + env_config.worker_index * num_envs_per_worker
        + env_config.vector_index
    )
    return Game(map_dir, engine_dir, server_port=port)


class NavigationEnv(gym.Env):
    ACTION_POOL = {
        "Move": [
            [(ActionVariable.WALK_SPEED, 0)],
            [(ActionVariable.WALK_SPEED, 5), (ActionVariable.WALK_DIR, 0)],
            [(ActionVariable.WALK_SPEED, 5), (ActionVariable.WALK_DIR, 90)],
            [(ActionVariable.WALK_SPEED, 5), (ActionVariable.WALK_DIR, 180)],
            [(ActionVariable.WALK_SPEED, 5), (ActionVariable.WALK_DIR, 270)],
        ],
        "Rotate": [
            [(ActionVariable.TURN_LR_DELTA, -2)],
            [(ActionVariable.TURN_LR_DELTA, 0)],
            [(ActionVariable.TURN_LR_DELTA, 2)],
        ],
    }
    TRIGGER_DISTANCE = 2

    metadata = {"render_modes": ["rgb_array"], "frames_per_second": 10}

    def __init__(self, env_config: EnvContext) -> None:
        super().__init__()
        self.config = env_config

        # build action and observation space
        self.action_space = spaces.Dict(
            {key: spaces.Discrete(len(val)) for key, val in self.ACTION_POOL.items()}
        )

        far = env_config["dmp_far"]
        width = env_config["dmp_width"]
        height = env_config["dmp_height"]

        self.observation_space = spaces.Dict(
            {
                "relative_pos": spaces.Box(low=-np.Inf, high=np.Inf, shape=(3,)),
                "depth_map": spaces.Box(low=0, high=far, shape=(height, width)),
            }
        )

        # build game backend and set game parameters
        game = create_game(env_config)
        game.set_map_id(env_config["map_id"])
        game.set_game_mode(Game.MODE_NAVIGATION)
        game.set_random_seed(env_config["random_seed"])
        game.set_episode_timeout(env_config["episode_timeout"])
        game.set_depth_map_size(width, height, far)
        game.turn_on_depth_map()
        game.init()
        self.game = game

        # initialize key variables
        self.target_location = None
        self.num_steps = 0

    def step(self, action):
        act = []
        for a_type, a_idx in action.items():
            act.extend(self.ACTION_POOL[a_type][a_idx])

        self.game.make_action_by_list({0: act})
        self.state = self.game.get_state()

        reward = 0
        done = self.game.is_episode_finished()

        pos = get_position(self.state)
        target_pos = self.game.get_target_location()

        if get_distance(pos, target_pos) <= self.TRIGGER_DISTANCE:
            done = True
            reward = 100

        self.num_steps += 1

        return self._get_obs(), reward, done, {}

    def reset(self):
        # reset game backend
        self.game.random_start_location()
        self.game.random_target_location()
        self.game.new_episode()

        # reset key variables
        self.target_location = self.game.get_target_location()
        self.num_steps = 0

        # get initial state
        self.state = self.game.get_state()

        return self._get_obs()

    def render(self, mode="rgb_array"):
        if mode != "rgb_array":
            raise NotImplementedError("Only support rgb_array mode!")

        far = self.game.get_depth_map_size()[-1]
        depth_map = self.state.depth_map
        img = (depth_map / far * 255).astype(np.uint8)
        h, w = img.shape
        img = cv2.resize(img, (w * self.render_scale, h * self.render_scale))
        return cv2.applyColorMap(img, cv2.COLORMAP_JET)

    def close(self) -> None:
        self.game.close()
        return super().close()

    def _get_obs(self):
        pos = np.asarray(get_position(self.state))
        target_pos = np.asarray(self.target_location)
        relative_pos = target_pos - pos

        return {
            "relative_pos": relative_pos,
            "depth_map": self.state.depth_map,
        }


if __name__ == "__main__":
    from ray.rllib.agents import ppo

    config = ppo.DEFAULT_CONFIG.copy()
    config.update(
        {
            "env": NavigationEnv,
            "env_config": {
                "map_dir": "../101_104",
                "engine_dir": "../wildscav-linux-backend-v1.0-benchmark",
                "random_seed": 123456,
                "dmp_far": 200,
                "dmp_width": 42,
                "dmp_height": 42,
                "episode_timeout": 30,
                "map_id": 1,
            },
            "num_workers": 1,
            "num_cpus_per_worker": 1,
            "framework": "torch",
        }
    )

    trainer = ppo.PPOTrainer(config)

    for i in range(10):
        result = trainer.train()
        print(result)
