import os, time
import random
import numpy as np
from typing import List

import gym
from gym.spaces import Box, MultiDiscrete, Tuple
from ray.rllib.env import EnvContext

from inspirai_fps.gamecore import ActionVariable, Game
from inspirai_fps.utils import get_distance, get_position


def standardization(data, axis=1):
    mu = np.mean(data, axis=axis)
    sigma = np.std(data, axis=axis)
    return (data - mu) / sigma


class SupplyGatherBaseEnv(gym.Env):
    """ 
    Base Gym Env for Supply Gathering,\\
    inherit this class to implement your own supply gathering environment,\\
    and implement the following methods:
        - _get_obs
        - _compute_reward
        - _action_process
    """

    def __init__(self, env_config: EnvContext):
        super().__init__()
        # set log
        cur_path = os.path.abspath(os.path.dirname(__file__))
        time_stamp = "Supply-%s" % time.strftime("%Y%m%d-%H%M%S")
        self.is_inference = env_config["inference"]
        
        self.server_port = 50052 + env_config.worker_index
        time_stamp += f"-{self.server_port}"
        self.log_path = os.path.expanduser("%s/%s" % (cur_path, time_stamp))
        print(f">>> New instance {self} on port: {self.server_port}")
        print(f"Worker Index: {env_config.worker_index}, VecEnv Index: {env_config.vector_index}")

        use_action_vars = [
            ActionVariable.WALK_DIR,
            ActionVariable.WALK_SPEED,
            ActionVariable.PICKUP,
        ]

        self.WALK_DIR_LIST = [0, 90, 180, 270]
        self.WALK_SPEED_LIST = [0, 8]  # [3, 6, 9] # [0, 1, 2]
        self.PICKUP_LIST = [True]

        self.action_space = MultiDiscrete(
            [
                len(self.WALK_DIR_LIST),
                len(self.WALK_SPEED_LIST),
                len(self.PICKUP_LIST),
            ]
        )

        self.supply_attribute_len = 3

        self.game = Game(
            map_dir=env_config["map_dir"],
            engine_dir=env_config["engine_dir"],
            server_port=self.server_port,
        )

        self.game.set_game_mode(Game.MODE_SUP_GATHER)
        self.game.set_available_actions(use_action_vars)
        self.game.set_map_id(env_config["map_id"])
        self.game.set_episode_timeout(env_config["timeout"])
        self.game.set_random_seed(env_config["random_seed"])
        self.game.set_supply_heatmap_center(env_config["heatmap_center"][0:2])
        self.game.set_supply_heatmap_radius(30)
        self.game.set_supply_indoor_richness(2)  # 10
        self.game.set_supply_outdoor_richness(2)  # 10
        self.game.set_supply_indoor_quantity_range(10, 20)
        self.game.set_supply_outdoor_quantity_range(1, 5)
        self.game.set_supply_spacing(2)

        self.is_inference = env_config.get("inference", False)
        self.turn_on_detailed_log = env_config["detailed_log"]
        self.args = env_config
        self.episode_count = 0

        self.target_supply_radius = 4  # heatmap center -> radius = 4, supply -> radius = 2

        self.list_spaces: List[gym.Space] = [Box(low=-np.Inf, high=np.Inf, shape=(3,), dtype=np.float32)]
        if env_config["use_depth_map"]:
            self.game.turn_on_depth_map()
            height = self.game.get_depth_map_height()
            width = self.game.get_depth_map_width()
            max_depth = self.game.get_depth_limit()
            self.list_spaces.append(Box(0, max_depth, (height, width), dtype=np.float32))
            self.observation_space = Tuple(self.list_spaces)
        else:
            self.observation_space = self.list_spaces[0]
        self.game.init()

    def reset(self):
        print("Reset for a new game ...")
        self.start_location = self._sample_start_location()
        self.game.set_start_location(self.start_location)
        self.episode_count += 1
        if self.args["record"] and self.episode_count % self.args["replay_interval"] == 0:
            self.game.turn_on_record()
            self.game.set_game_replay_suffix(self.args["replay_suffix"])
        else:
            self.game.turn_off_record()
        self.game.new_episode()
        state = self.game.get_state()

        self.collected_supply = 0
        self.running_steps = 0
        self.episode_reward = 0
        self.valid_collected_supply = 0  # number of valid collected supply
        # self.target_supply_flag = False # whether the agent still exists
        return state

    def _sample_start_location(self):
        loc = self.game.get_valid_locations()
        out_loc = loc["outdoor"]
        start_loc = random.choice(out_loc)
        return start_loc

    def _action_process(self, action):
        walk_dir = self.WALK_DIR_LIST[action[0]]
        walk_speed = self.WALK_SPEED_LIST[action[1]]
        pickup = self.PICKUP_LIST[action[2]]

        return {0: [walk_dir, walk_speed, pickup]}

    def step(self, action):
        """
        Parameters
        ----------
        action : list of action values

        Procedure
        ----------
        1. process action to cmd and then backend env execute action
        2. get new state from backend env
        3. compute reward from new state
        4. process new state to get the new observation
        """

        self.running_steps += 1
        # 执行动作
        action_cmd = self._action_process(action)
        self.game.make_action(action_cmd)

        # 状态转移
        state = self.game.get_state()

        # 计算 reward
        # _compute_reward = getattr(self, '_compute_reward')
        reward = self._compute_reward(state, action_cmd)
        self.episode_reward += reward
 

        # 计算 state
        # _get_obs = getattr(self, '_get_obs')
        self.curr_obs = self._get_obs(state)

        done = self.game.is_episode_finished()

        if done:
            print(f"{self.valid_collected_supply=},{state.num_supply}")





        return self.curr_obs, reward, done, {}

    def _get_obs(self, state):
        """
        method to process state to get observation

        Parameters
        ----------
        state: AgentState object got from backend env
        """
        raise NotImplementedError()

    def _compute_reward(self, state, action):
        """reward process method

        Parameters
        ----------
        state: AgentState object got from backend env
        action: action list got from agent
        """
        raise NotImplementedError()


class SupplyGatherDiscreteSingleTarget(SupplyGatherBaseEnv):
    """
    Supply Gathering Env with discrete action space

    Task Design
    ----------
    The agent is randomly spawned near the supply heatmap center with the following goals:
    1. reach the supply heatmap center
    2. collect supplies in the world.

    Observation Space
    ----------
    `obs`: normalized direction vector pointing to the goal location

    Reward Shaping
    ----------
    1. successfully collect supply: 300
    2. punish for moving away from supply, award for moving towards supply
    3. punish for taking pickup action but not collect supply successfully
    4. punish for single step movement

    Note
    ----------
    Different from `SupplyGatherDiscreteSingleTargetTwo`,
    the observation in `SupplyGatherDiscreteSingleTarget` directly provides the agent with direction information,
    making it easy to learn, thus reward does not need distance as guidance.
    """

    def __init__(self, env_config: EnvContext):
        super().__init__(env_config)

    def reset(self):
        state = super().reset()

        # the initial goal is the supply heatmap center
        self.target_supply = [
            self.args["heatmap_center"][0],
            self.args["heatmap_center"][1],
            self.args["heatmap_center"][2],
        ]
        obs = []
        cur_pos = np.asarray(get_position(state))
        tar_pos = np.asarray(self.target_supply)
        dir_vec = tar_pos - cur_pos
        
        return dir_vec

    def _other_process(self, done: bool):
        # if found no good solution, stop the episode
        if (self.cur_distance >= 10 and 0 <= self.valid_collected_supply < 10) or (self.cur_distance >= 15 and 10 <= self.valid_collected_supply < 50):
            done = True
        return done

    def _compute_reward(self, state, action_cmd):
        reward = 0
        if self.running_steps == 1:
            return reward
        if not self.game.is_episode_finished():
            # movement punishment
            # reward -= 1

            # punish for moving away from supply and award for moving towards supply
            self.cur_distance = get_distance(
                [self.target_supply[0], self.target_supply[1], self.target_supply[2]],
                get_position(state),
            )
            # reward += (self.target_supply_radius - self.cur_distance) * 5

            # reaching the initial goal (supply heatmap center)
            if self.valid_collected_supply == 0 and self.cur_distance <= self.target_supply_radius:
                reward += 1
                self.target_supply = None
                self.valid_collected_supply += 1
                self.target_supply_radius = 4  # 第一个目标完成，修改物资半径为2

            # reaching the second goal (successfully collect supplies)
            if state.num_supply > self.collected_supply and self.cur_distance <= 1:
                reward +=1
                self.target_supply = None
                self.valid_collected_supply += 1

            self.collected_supply = state.num_supply

        return reward

    def _get_obs(self, state):
        # get supply info of all nearby supplies
        self.np_supply_states = [
                [
                    supply.position_x,
                    supply.position_y,
                    supply.position_z,
                ]
            for supply in state.supply_states.values()
        ]

        # reinitialize: get target information
        supply_distances = [get_distance([supply[0], supply[1], supply[2]], get_position(state)) for supply in self.np_supply_states]
        if self.target_supply is None:
            # target supply is the closest supply
            if supply_distances:
                self.target_supply = self.np_supply_states[supply_distances.index(min(supply_distances))]
            else:
                # if no supply nearby, the target supply is set to be the supply heatmap center
                temp = self.args["heatmap_center"].copy()
                self.target_supply = temp
            # get distance to target supply
            self.cur_distance = get_distance(self.target_supply, get_position(state)) if self.target_supply is not None else None
        else:
            if self.target_supply==self.args["heatmap_center"]:
                if self.np_supply_states:
                    self.target_supply = self.np_supply_states[supply_distances.index(min(supply_distances))]

        cur_pos = np.asarray(get_position(state))
        tar_pos = np.asarray([self.target_supply[0], self.target_supply[1], self.target_supply[2]]) if self.target_supply is not None else np.asarray(self.args["heatmap_center"])
        dir_vec = tar_pos - cur_pos

        return dir_vec

