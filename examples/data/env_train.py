import cv2
import random
from typing import Dict

import gym
import numpy as np
from gym import spaces
from ray.rllib.env import EnvContext
from inspirai_fps.utils import get_distance, get_position
from inspirai_fps.gamecore import Game
from inspirai_fps.gamecore import ActionVariable as A

BASE_WORKER_PORT = 50000


class NavigationEnv(gym.Env):
    def __init__(self, config: EnvContext):
        self.config = config
        self.render_scale = config.get("render_scale", 1)

        env_seed = config.get("random_seed", 0) + config.worker_index

        # only 240 * 320 can be aotu transform into conv model
        dmp_width = config["dmp_width"]
        dmp_height = config["dmp_height"]
        dmp_far = config["dmp_far"]

        obs_space_1 = spaces.Box(low=-np.Inf, high=np.Inf, shape=(3,), dtype=np.float32)
        obs_space_2 = spaces.Box(
            low=0, high=dmp_far, shape=(dmp_height, dmp_width), dtype=np.float32
        )
        self.observation_space = spaces.Tuple([obs_space_1, obs_space_2])
        # self.observation_space = obs_space_1
        self.action_dict = {
            "move": [
                [(A.WALK_DIR, 0), (A.WALK_SPEED, 0)],
                [(A.WALK_DIR, 0), (A.WALK_SPEED, 8)],
                [(A.WALK_DIR, 90), (A.WALK_SPEED, 8)],
                [(A.WALK_DIR, 180), (A.WALK_SPEED, 8)],
                [(A.WALK_DIR, 270), (A.WALK_SPEED, 8)],
                # [(A.JUMP, True)],
            ],
            # "jump": [
            #     [(A.JUMP, False)],
            #     [(A.JUMP, True)],
            # ],
            "turn_lr": [
                [(A.TURN_LR_DELTA, -2)],
                [(A.TURN_LR_DELTA, -1)],
                [(A.TURN_LR_DELTA, 0)],
                [(A.TURN_LR_DELTA, 1)],
                [(A.TURN_LR_DELTA, 2)],
            ],
            # "look_ud": [
            #     [(A.LOOK_UD_DELTA, -2)],
            #     [(A.LOOK_UD_DELTA, -1)],
            #     [(A.LOOK_UD_DELTA, 0)],
            #     [(A.LOOK_UD_DELTA, 1)],
            #     [(A.LOOK_UD_DELTA, 2)],
            # ],
        }

        self.action_space = spaces.Dict({
            k: spaces.Discrete(len(v)) for k, v in self.action_dict.items()
        })

        self.replay_suffix = config.get("replay_suffix", "")
        self.print_log = config.get("detailed_log", False)
        self.seed(env_seed)
        self.server_port = (
                config.get("base_worker_port", BASE_WORKER_PORT) + config.worker_index
        )
        if self.config.get("in_evaluation", False):
            self.server_port += 100
        print(f">>> New instance {self} on port: {self.server_port}")
        print(
            f"Worker Index: {config.worker_index}, VecEnv Index: {config.vector_index}"
        )

        self.game = Game(
            map_dir=config["map_dir"],
            engine_dir=config["engine_dir"],
            server_port=self.server_port,
        )
        self.game.set_map_id(config["map_id"])
        self.game.set_episode_timeout(config["timeout"])
        self.game.set_random_seed(env_seed)
        self.start_location = config.get("start_location", [0, 0, 0])
        if self.config.get("record", False):
            self.game.turn_on_record()
        # self.game.turn_on_depth_map()
        self.game.set_game_replay_suffix(self.replay_suffix)
        self.game.set_game_mode(Game.MODE_NAVIGATION)
        self.game.set_depth_map_size(dmp_width, dmp_height, far=dmp_far)
        self.target_location = config.get("target_location", [0, 0, 0])
        self.start_loc = config.get("start_location", [0, 0, 0])
        self.game.set_target_location(self.target_location)

        # 101 - [x(-100,100) z(-100,100)]
        # 102 - [x(-25,175) z(-50,150)]
        # 103 - [x(-100,0) z(-50,50)]
        # 104 - [x(-125,-25) z(-110,-30)]
        self.map_select = {
            101: [[-100, 100], [-100, 100]],
            102: [[-25, 175], [-50, 150]],
            103: [[-100, 0], [-50, 50]],
            104: [[-125, -25], [-110, -30]],
        }
        limit = self.map_select.get(config["map_id"], [[-500, 500], [-500, 500]])
        locations = self.game.get_valid_locations()

        def in_map(loc):
            return limit[0][0] <= loc[0] <= limit[0][1] and limit[1][0] <= loc[2] <= limit[1][1]

        self.indoor_loc = list(filter(in_map, locations["indoor"]))
        self.outdoor_loc = list(filter(in_map, locations["outdoor"]))

        self.loc_20 = []
        self.loc_50 = []
        self.loc_80 = []
        for loc in self.outdoor_loc:
            if get_distance(loc, self.target_location) <= 20:
                self.loc_20.append(loc)
            elif get_distance(loc, self.target_location) <= 40:
                self.loc_50.append(loc)
            elif get_distance(loc, self.target_location) <= 60:
                self.loc_80.append(loc)

        self.limit = 100

        self.game.init()
        self.episodes = 0
        self.episode_reward = 0

    def _get_obs(self):
        cur_pos = np.asarray(get_position(self.state))
        tar_pos = np.asarray(self.target_location)
        # self.state.depth_map.copy()
        return tar_pos - cur_pos
            

    def step(self, action):
        # action = self._action_process(action_idxs)
        # self.game.make_action({0: action})
        action_list = self._action_process(action)
        self.game.make_action_by_list({0: action_list})
        state = self.game.get_state()
        done = self.game.is_episode_finished()

        self.running_steps += 1
        cur_pos = get_position(state)
        tar_pos = self.target_location
        # reward = -get_distance(cur_pos, tar_pos)
        reward = get_distance(get_position(self.state),tar_pos)-get_distance(cur_pos,tar_pos)
        self.state = state
        if get_distance(cur_pos, tar_pos) <= 2:
            reward += 100
            done = True
        # if get_distance(cur_pos, tar_pos) >= self.limit * 1.5:
        #     done = True
        #     reward = -100

        if done:
            if self.print_log:
                Start = np.round(np.asarray(self.start_loc), 2).tolist()
                Target = np.round(np.asarray(self.target_location), 2).tolist()
                End = np.round(np.asarray(get_position(self.state)), 2).tolist()
                Step = self.running_steps
                Reward = reward
                print(f"{Start=}\t{Target=}\t{End=}\t{Step=}\t{Reward=}")
            self.episode_reward += reward

        return self._get_obs(), reward, done, {}

    def reset(self):

        print("Reset for a new game ...")

        # if self.episodes <= 500:
        #     self.start_loc = random.choice(self.loc_20)
        #     # self.game.set_episode_timeout(60)
        # elif self.episodes <= 1000:
        #     self.start_loc = random.choice(self.loc_50)
        #     # self.game.set_episode_timeout(120)

        # elif self.episodes <= 2000:
        #     self.start_loc = random.choice(self.loc_80)
        #     # self.game.set_episode_timeout(180)
        # else:
        #     self.start_loc = random.choice(self.outdoor_loc)
        #     self.game.set_episode_timeout(300)

        # # if self.config.get("in_evaluation",False):
        self.state = self.game.get_state()
        self.start_loc = random.choice(self.outdoor_loc)

        self.limit = get_distance(self.target_location, self.start_loc)

        self.game.set_start_location(self.start_loc)

        self.game.new_episode()
        self.state = self.game.get_state()

        self.running_steps = 0
        self.episodes += 1
        return self._get_obs()

    def close(self):
        self.game.close()
        return super().close()

    def _action_process(self, action: Dict[str, int]):
        action_list = []
        for action_name, action_idx in action.items():
            action_list.extend(self.action_dict[action_name][action_idx])
        return action_list

    def render(self, mode="rgb_array"):
        if mode != "rgb_array":
            raise NotImplementedError("Only support rgb_array mode!")

        far = self.game.get_depth_map_size()[-1]
        depth_map = self.state.depth_map
        img = (depth_map / far * 255).astype(np.uint8)
        h, w = img.shape
        img = cv2.resize(img, (w * self.render_scale, h * self.render_scale))
        return cv2.applyColorMap(img, cv2.COLORMAP_JET)
