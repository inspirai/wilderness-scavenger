import gym
import numpy as np
from gym import spaces
from ray.rllib.env import EnvContext
from inspirai_fps.utils import get_distance, get_position
from inspirai_fps.gamecore import Game, ActionVariable
import os
import cv2

BASE_WORKER_PORT = 50000


class BaseEnv(gym.Env):
    def __init__(self, config: EnvContext):
        super().__init__()

        self.config = config
        self.record = config.get("record", False)
        self.replay_suffix = config.get("replay_suffix", "")
        self.print_log = config.get("detailed_log", False)

        self.seed(config["random_seed"])
        self.server_port = BASE_WORKER_PORT + config.worker_index
        print(f">>> New instance {self} on port: {self.server_port}")
        print(f"Worker Index: {config.worker_index}, VecEnv Index: {config.vector_index}")

        self.game = Game(map_dir=config["map_dir"], engine_dir=config["engine_dir"], server_port=self.server_port)
        self.game.set_map_id(config["map_id"])
        self.game.set_episode_timeout(config["timeout"])
        self.game.set_random_seed(config["random_seed"])
        self.start_location = config.get("start_location", [0, 0, 0])
        if self.record:
            self.game.turn_on_record()
        else:
            self.game.turn_off_record()
        self.game.set_game_replay_suffix(self.replay_suffix)


    def reset(self):
        print("Reset for a new game ...")
        self._reset_game_config()

        self.game.new_episode()
        self.state = self.game.get_state()
        self.running_steps = 0
        return self._get_obs()

    def close(self):
        self.game.close()
        return super().close()

    def render(self, mode="replay"):
        if not self.config.get("use_depth",None):
            raise Exception("You do not turn on the use-depth-map parameter,"
                            "so there is no chance of visualizing depth-map for you")
        
        SCALE = 1
        WIDTH,HEIGHT,FAR = self.game.get_depth_map_size()
            
        def visualize(map_id, ts, depth_map):
            img = (depth_map / FAR * 255).astype(np.uint8)
            h, w = img.shape
            img = cv2.resize(img, (w * SCALE, h * SCALE))
            img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
            cv2.imwrite(f"dmp/depth_map={map_id}_ts={ts}.png", img)
            
        os.makedirs("dmp", exist_ok=True)
        depth_map = self.state.depth_map
        visualize(self.config["map_id"], self.game.get_time_step(), depth_map)




    def _reset_game_config(self):
        raise NotImplementedError()

    def _get_obs(self):
        raise NotImplementedError()


class NavigationBaseEnv(BaseEnv):
    def __init__(self, config: EnvContext):
        super().__init__(config)

        self.start_range = config["start_range"]
        self.start_hight = config["start_hight"]
        self.trigger_range = self.game.get_target_reach_distance()
        self.target_location = config["target_location"]

        self.game.set_game_mode(Game.MODE_NAVIGATION)
        self.game.set_target_location(self.target_location)

        self.action_pools = {
            ActionVariable.WALK_DIR: [0, 90, 180, 270],
            ActionVariable.WALK_SPEED: [3, 6],
        }
        self.action_space = spaces.MultiDiscrete([len(pool) for pool in self.action_pools.values()])
        self.observation_space = spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)

        self.game.set_available_actions([action_name for action_name in self.action_pools.keys()])
        self.game.init()

    def _reset_game_config(self):
        self.start_location = self._sample_start_location()
        self.game.set_start_location(self.start_location)

    def step(self, action):
        action_cmd = self._action_process(action)
        self.game.make_action({0: action_cmd})
        self.state = self.game.get_state()
        done = self.game.is_episode_finished()
        
        self.running_steps += 1
        cur_pos = get_position(self.state)
        tar_pos = self.target_location
        reward = -get_distance(cur_pos, tar_pos)

        if get_distance(cur_pos, tar_pos) <= self.trigger_range:
            reward += 100
            done = True
        if done:
            if self.print_log:
                Start = np.round(np.asarray(self.start_location), 2).tolist()
                Target = np.round(np.asarray(self.target_location), 2).tolist()
                End =  np.round(np.asarray(get_position(self.state)), 2).tolist()
                Step = self.running_steps
                Reward = reward
                print(f"{Start=}\t{Target=}\t{End=}\t{Step=}\t{Reward=}")

        return self._get_obs(), reward, done, {}


    def _get_obs(self):
        cur_pos = np.asarray(get_position(self.state))
        tar_pos = np.asarray(self.target_location)
        dir_vec = tar_pos - cur_pos
        return dir_vec / np.linalg.norm(dir_vec)

    def _action_process(self, action):
        action_values = list(self.action_pools.values())
        return [action_values[i][action[i]] for i in range(len(action))]

    def _sample_start_location(self):
        angle = np.random.uniform(0, 360)
        distance_to_trigger = abs(np.random.normal(scale=self.start_range))
        vec_len = self.trigger_range + distance_to_trigger
        dx = np.sin(angle) * vec_len
        dz = np.cos(angle) * vec_len
        x = self.target_location[0] + np.random.randint(-30,30)
        z = self.target_location[2] + np.random.randint(-30,30)
        return [x, self.start_hight, z]


