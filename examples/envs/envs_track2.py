import os, time
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
        if self.is_inference:
            self.server_port = 50052
            time_stamp += f"-{self.server_port}"
            self.log_path = os.path.expanduser("%s/%s" % (cur_path, time_stamp))
            with open(self.log_path + "log.txt", "w") as f:
                f.write(f">>> {self.__class__}, log:\n")
        else:
            self.server_port = 50052 + env_config.worker_index
            time_stamp += f"-{self.server_port}"
            self.log_path = os.path.expanduser("%s/%s" % (cur_path, time_stamp))
            print(f">>> New instance {self} on port: {self.server_port}")
            print(f"Worker Index: {env_config.worker_index}, VecEnv Index: {env_config.vector_index}")
            with open(self.log_path + "log.txt", "w") as f:
                f.write(f">>> {self.__class__}, server_port: {self.server_port} , worker_index: {env_config.worker_index}, log:\n")

        use_action_vars = [
            ActionVariable.WALK_DIR,
            ActionVariable.WALK_SPEED,
            ActionVariable.PICKUP,
        ]

        self.WALK_DIR_LIST = [0, 45, 90, 135, 180, 225, 270, 315]
        self.WALK_SPEED_LIST = [3, 6, 9]  # [3, 6, 9] # [0, 1, 2]
        self.PICKUP_LIST = [True, False]

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
        self.game.set_supply_heatmap_center(env_config["heatmap_center"])
        self.game.set_supply_heatmap_radius(30)
        self.game.set_supply_indoor_richness(2)  # 10
        self.game.set_supply_outdoor_richness(2)  # 10
        self.game.set_supply_indoor_quantity_range(10, 20)
        self.game.set_supply_outdoor_quantity_range(1, 5)
        self.game.set_supply_spacing(1)

        self.game.set_time_scale(env_config["time_scale"])
        self.is_inference = env_config["inference"] if "inference" in env_config else False
        self.turn_on_detailed_log = env_config["detailed_log"]
        self.args = env_config
        self.episode_count = 0

        self.target_supply_radius = 4  # heatmap center -> radius = 4, supply -> radius = 2

        self.list_spaces: List[gym.Space] = [Box(low=-1, high=1, shape=(3,), dtype=np.float32)]
        if env_config["use_depth"]:
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
            self.game.set_game_replay_suffix(self.args["replay_name"])
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
        angle = np.random.uniform(0, 360)
        distance_to_trigger = abs(np.random.normal(scale=self.args["start_range"]))
        vec_len = 1 + distance_to_trigger
        # vec_len = self.game.trigger_range + np.random.uniform(0, self.start_range)
        dx = np.sin(angle) * vec_len
        dz = np.cos(angle) * vec_len
        x = self.args["heatmap_center"][0] + dx
        z = self.args["heatmap_center"][1] + dz
        return [x, self.args["start_hight"], z]

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
        if self.turn_on_detailed_log:
            with open(self.log_path + "log.txt", "a") as f:
                f.write(f"\nstep:{self.running_steps};\t")
                f.write(f"动作：{action_cmd};\t")
                f.write(f"奖励:{round(reward, 2)};\t")

        # 计算 state
        # _get_obs = getattr(self, '_get_obs')
        self.curr_obs = self._get_obs(state)

        done = self.game.is_episode_finished()

        _other_process = getattr(self, "_other_process")
        done = _other_process(done)

        if done:
            with open(self.log_path + "log.txt", "a") as f:
                f.write(f"\nepisode总共走了这么多步：{self.running_steps}\n")
                f.write(f"捡到的supply总量：{self.collected_supply}\n")
                f.write(f"总奖励：{self.episode_reward}\n")
                f.write(f"有效supply总量：{self.valid_collected_supply}\n")

            if self.is_inference:
                self.game.close()

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
            0,
            self.args["heatmap_center"][1],
        ]
        obs = []
        cur_pos = np.asarray(get_position(state))
        tar_pos = np.asarray(self.target_supply)
        dir_vec = tar_pos - cur_pos
        dir_vec = dir_vec / np.linalg.norm(dir_vec)
        obs.append(dir_vec.tolist())

        if self.args["use_depth"]:
            obs.append(state.depth_map.tolist())
            return obs
        else:
            return obs[0]

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
            reward -= 1

            # punish for taking PICKUP but with no increase in #supply
            if action_cmd[0][2] == True and state.num_supply == self.collected_supply:
                reward -= 50

            # punish for moving away from supply and award for moving towards supply
            self.cur_distance = get_distance(
                [self.target_supply[0], self.target_supply[1], self.target_supply[2]],
                get_position(state),
            )
            reward += (self.target_supply_radius - self.cur_distance) * 5

            # reaching the initial goal (supply heatmap center)
            if self.valid_collected_supply == 0 and self.cur_distance <= self.target_supply_radius:
                reward += 300
                self.target_supply = None
                self.valid_collected_supply += 1
                self.target_supply_radius = 4  # 第一个目标完成，修改物资半径为2

            # reaching the second goal (successfully collect supplies)
            if state.num_supply > self.collected_supply and self.cur_distance <= 1:
                reward += 300
                self.target_supply = None
                self.valid_collected_supply += state.num_supply - self.collected_supply

            self.collected_supply = state.num_supply

        return reward

    def _get_obs(self, state):
        # get supply info of all nearby supplies
        self.np_supply_states = [
            np.asarray(
                [
                    supply.position_x,
                    supply.position_y,
                    supply.position_z,
                    supply.quantity,
                ]
            )
            for supply in state.supply_states
        ]

        # reinitialize: get target information
        if self.target_supply is None:
            supply_distances = [get_distance([supply[0], supply[1], supply[2]], get_position(state)) for supply in self.np_supply_states]
            # target supply is the closest supply
            if supply_distances:
                self.target_supply = self.np_supply_states[supply_distances.index(min(supply_distances))]
            else:
                # if no supply nearby, the target supply is set to be the supply heatmap center
                temp = self.args["heatmap_center"].copy()
                temp.append(-1)
                self.target_supply = temp
            # get distance to target supply
            self.cur_distance = get_distance(self.target_supply[:-1], get_position(state)) if self.target_supply is not None else None
        else:
            self.cur_distance = None
            if self.target_supply is not None:
                self.cur_distance = get_distance(
                    [
                        self.target_supply[0],
                        self.target_supply[1],
                        self.target_supply[2],
                    ],
                    get_position(state),
                )

        self._write_obs_log(state, self.cur_distance)

        cur_pos = np.asarray(get_position(state))
        tar_pos = np.asarray([self.target_supply[0], self.target_supply[1], self.target_supply[2]]) if self.target_supply is not None else np.asarray(self.args["heatmap_center"])
        dir_vec = tar_pos - cur_pos
        dir_vec = dir_vec / np.linalg.norm(dir_vec)
        obs = []
        obs.append(dir_vec.tolist())

        if self.args["use_depth"]:
            obs.append(state.depth_map.tolist())
            return obs
        else:
            return obs[0]

    def _write_obs_log(self, state, cur_distance):
        if self.turn_on_detailed_log:
            with open(self.log_path + "log.txt", "a") as f:
                f.write(f"CurrentLocation: {state.position_x:.2f}, {state.position_y:.2f}, {state.position_z:.2f};\t")
                f.write(f"NearbySupply: {len(state.supply_states)};\t")
                f.write(f"CollectedSupply: {state.num_supply};\t")
                f.write(f"Target: {[round(self.target_supply[loc_index], 2) for loc_index in range(3)] if self.target_supply is not None else None};\t")
                f.write(f"Distance: {round(cur_distance, 2) if cur_distance is not None else None};\t")


class SupplyGatherDiscreteEasySingleTargetVision(SupplyGatherDiscreteSingleTarget):
    """在 `SupplyGatherDiscreteSingleTarget` 基础上加入了 `Vision`"""

    def __init__(self, env_config: EnvContext):
        super().__init__(env_config)

    def _compute_reward(self, state, action):
        # 引入 视野 奖励
        reward = 0
        if self.running_steps == 1:
            return reward
        if not self.game.is_episode_finished():
            # 移动惩罚
            reward -= 1

            # 远离目标惩罚，接近目标奖励
            self.cur_distance = get_distance(
                [self.target_supply[0], self.target_supply[1], self.target_supply[2]],
                get_position(state),
            )
            reward += (self.target_supply_radius + 2 - self.cur_distance) * 40

            # 到达热力图中心附近（第一个目标）
            if self.valid_collected_supply == 0 and self.cur_distance <= self.target_supply_radius:
                reward += 300
                self.target_supply = None
                self.valid_collected_supply += 1

            # 捡到 目标物资 给奖励
            if state.num_supply > self.collected_supply and self.cur_distance <= 1:
                reward += 300
                self.target_supply = None
                self.valid_collected_supply += state.num_supply - self.collected_supply

            self.collected_supply = state.num_supply

        return reward

    def _get_obs(self, state):
        # TODO obs 加入 视野
        # 当前附近的所有物资
        self.np_supply_states = [
            np.asarray(
                [
                    supply.position_x,
                    supply.position_y,
                    supply.position_z,
                    supply.quantity,
                ]
            )
            for supply in state.supply_states
        ]

        # 重新初始化 目标物资
        if self.target_supply is None:
            supply_distance = [get_distance([supply[0], supply[1], supply[2]], get_position(state)) for supply in self.np_supply_states]
            # 把距离最近的 supply 设置为 target
            self.target_supply = self.np_supply_states[supply_distance.index(min(supply_distance))] if len(supply_distance) != 0 else None  # self.args["heatmap_center"]
            # 初始化当前所在位置到 supply 的距离
            self.cur_distance = get_distance(self.target_supply[:-1], get_position(state)) if self.target_supply is not None else None
        else:
            self.cur_distance = (
                get_distance(
                    [
                        self.target_supply[0],
                        self.target_supply[1],
                        self.target_supply[2],
                    ],
                    get_position(state),
                )
                if self.target_supply is not None
                else None
            )

        self._write_obs_log(state, self.cur_distance)

        cur_pos = np.asarray(get_position(state))
        tar_pos = np.asarray([self.target_supply[0], self.target_supply[1], self.target_supply[2]]) if self.target_supply is not None else np.asarray(self.args["heatmap_center"])
        dir_vec = tar_pos - cur_pos
        obs = dir_vec / np.linalg.norm(dir_vec)
        return obs


class SupplyGatherDiscreteSingleTargetTwo(SupplyGatherBaseEnv):
    """智能体随机出生在物资热力图中心附近，目标是用最少的步数不断地采集距离它最近的目标物资
        state: (1) 当前位置; (2) 目标物资位置；（把最近的一个物资作为目标，捡到目标后才会重新设置目标）
        reward: (1) 捡到目标物资奖励; (2) 远离目标物资惩罚, 接近目标物资奖励;(last_distance - cur_distance)；
        (3) 未捡到目标物资而做出pickup惩罚; (4) 移动惩罚；


    Note: 与 `SupplyGatherDiscreteSingleTarget` 不同的是，`SupplyGatherDiscreteSingleTargetTwo` 的 state 只给智能体提供了两个位置坐标，
    智能体需要学会隐含的距离信息， reward 使用 diff in distance 来引导智能体快速学习。
    """

    def __init__(self, env_config: EnvContext):
        super().__init__(env_config)

        # 智能体可以看见的目标物资的数量
        self.visible_supply_max_len = 1  # 10
        # 物资的属性的长度
        self.supply_attribute_len = 4
        # 智能体位置的取值区间
        low = [np.asarray([-np.inf] * 4, dtype=np.float32)]
        high = [np.asarray([np.inf] * 4, dtype=np.float32)]
        # 物资位置的取值区间
        low.extend([np.asarray([-np.inf] * self.supply_attribute_len, dtype=np.float32) for _ in range(self.visible_supply_max_len)])
        high.extend([np.asarray([np.inf] * self.supply_attribute_len, dtype=np.float32) for _ in range(self.visible_supply_max_len)])
        self.observation_space = Box(
            low=np.asarray(low),
            high=np.asarray(high),
            shape=(self.visible_supply_max_len + 1, self.supply_attribute_len),
            dtype=np.float32,
        )

    def reset(self):
        state = super().reset()
        self.target_supply_flag = False  # 目标物资是否还在；True还在
        obs = [np.asarray([state.position_x, state.position_y, state.position_z, 0])]  # 最后一个属性是 state.allow_pickup
        obs.extend([np.zeros(self.supply_attribute_len)])

        return np.asarray(obs)

    def step(self, action):
        self.running_steps += 1
        # 执行动作
        action_cmd = self._action_process(action)
        self.game.make_action({0: action_cmd})
        if self.turn_on_detailed_log:
            with open(self.log_path + "log.txt", "a") as f:
                f.write(f"\nstep:{self.running_steps};\t")
                f.write(f"动作：{action_cmd};\t")

        # 状态转移
        state = self.game.get_state()
        self.curr_obs = self._get_obs(state, action_cmd)

        done = self.game.is_episode_finished()

        done = self._other_process(done)

        if done:
            with open(self.log_path + "log.txt", "a") as f:
                f.write(f"\nepisode总共走了这么多步：{self.running_steps}\n")
                f.write(f"捡到的supply总量：{self.collected_supply}\n")
                f.write(f"总奖励：{self.episode_reward}\n")
                f.write(f"有效supply总量：{self.valid_collected_supply}\n")

            if self.is_inference:
                self.game.close()

        return self.curr_obs, self.reward, done, {}

    def _compute_reward(self, state, action_cmd):
        reward = 0
        if not self.game.is_episode_finished():
            # 移动惩罚
            reward += -1

            # 如果做出 pickup 但是 supply 数量没有增加则惩罚
            # if action_cmd[2] == True and self.target_supply_flag == True:
            #     reward -= 300

            # 没有目标，返回 0 奖励
            if hasattr(self, "cur_distance") and self.cur_distance is None:
                return 0

            # 捡到目标物资奖励
            if state.num_supply > self.collected_supply and hasattr(self, "target_supply") and self.cur_distance < 1:
                reward += 500
                # done = True
                self.target_supply = None
                self.target_supply_flag = False
                self.valid_collected_supply += 1

            self.collected_supply = state.num_supply

            # 接近目标物资奖励，远离目标物资惩罚
            if self.running_steps >= 1 and self.target_supply is not None:
                reward += (self.last_distance - self.cur_distance) * 1000  # if self.last_distance == -1 else
                self.last_distance = self.cur_distance

        return reward

    def _get_obs(self, state, action_cmd):
        obs = [np.asarray([state.position_x, state.position_y, state.position_z, 0])]  # 最后一个是 state.allow_pickup

        # 当前附近的所有物资
        self.np_supply_states = [
            np.asarray(
                [
                    supply.position_x,
                    supply.position_y,
                    supply.position_z,
                    supply.quantity,
                ]
            )
            for supply in state.supply_states
        ]

        # 初始化目标物资
        if self.running_steps == 1:
            supply_distance = [get_distance([supply[0], supply[1], supply[2]], get_position(state)) for supply in self.np_supply_states]
            # 把距离最近的 supply 设置为 target
            self.target_supply = self.np_supply_states[supply_distance.index(min(supply_distance))] if len(supply_distance) != 0 else None
            # 初始化当前所在位置到 target supply 的距离
            self.last_distance = get_distance(self.target_supply[:-1], get_position(state)) if self.target_supply is not None else None

        # 判断目标物资是否已经被捡走; True还在，False捡走
        if self.target_supply_flag is not None and self.target_supply is not None:
            self.target_supply_flag = False if state.num_supply > self.collected_supply and get_distance(self.target_supply[:-1], get_position(state)) <= 1 else True
        else:
            self.target_supply_flag = False

        self.cur_distance = get_distance(self.target_supply[:-1], get_position(state)) if self.target_supply is not None else None

        # 计算奖励
        self.reward = self._compute_reward(state, action_cmd)
        self.episode_reward += self.reward

        # 如果目标物资被捡走，则重新设置目标
        if not self.target_supply_flag:
            if len(self.np_supply_states) == 0:
                self.target_supply = None
                self.last_distance = -1
            else:
                supply_distance = [get_distance([supply[0], supply[1], supply[2]], get_position(state)) for supply in self.np_supply_states]
                self.target_supply = self.np_supply_states[supply_distance.index(min(supply_distance))]
                self.last_distance = get_distance(self.target_supply[:-1], get_position(state))

        self._write_obs_log(state, self.cur_distance)

        # 构造 obs
        if self.running_steps < 1:
            obs.extend([np.zeros(self.supply_attribute_len)])
        elif self.running_steps >= 1:
            if len(self.np_supply_states) >= self.visible_supply_max_len:
                obs.extend([self.target_supply])
            else:
                obs.extend([np.zeros(self.supply_attribute_len)])
                # obs.extend([np.zeros(self.supply_attribute_len)] * (self.visible_supply_max_len - len(self.init_supply_states)))

        return np.asarray(obs)

    def _action_process(self, action):
        walk_dir = self.WALK_DIR_LIST[action[0]]
        walk_speed = self.WALK_SPEED_LIST[action[1]]
        pickup = self.PICKUP_LIST[action[2]]

        return [walk_dir, walk_speed, pickup]

    def _other_process(self, done: bool):
        # 如果智能体在探索过程中逐渐陷入无解，则手动终止本局，进入下一局
        if (self.cur_distance >= 10 and 0 <= self.valid_collected_supply < 10) or (self.cur_distance >= 15 and 10 <= self.valid_collected_supply < 50):
            done = True
        return done

    def _write_obs_log(self, state, cur_distance):
        if self.turn_on_detailed_log:
            with open(self.log_path + "log.txt", "a") as f:
                f.write(f"奖励:{round(self.reward, 2)};\t")
                f.write(f"当前位置：{state.position_x:.2f}, {state.position_y:.2f}, {state.position_z:.2f};\t")
                f.write(f"附近物资：{len(state.supply_states)};\t")  # {state.supply_states}
                f.write(f"收集到的物资：{state.num_supply};\t")
                f.write(f"目标：{[round(self.target_supply[loc_index], 2) for loc_index in range(3)] if self.target_supply is not None else None};\t")
                f.write(f"距离目标：{round(cur_distance, 2) if cur_distance is not None else None};\t")
