import os
import sys
import tarfile

import gym
import numpy as np
from gym import spaces
from ray.rllib.env import EnvContext
from ray.rllib.env.multi_agent_env import MultiAgentEnv
import time
import random
import argparse

from rich.progress import track
from rich.console import Console
console = Console()

from inspirai_fps.gamecore import Game, ActionVariable
from inspirai_fps.utils import get_position,get_distance

BASE_WORKER_PORT = 50000

from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser()

# game setup
parser.add_argument("--timeout", type=int, default=60 * 2)  # The time length of one game (sec)
parser.add_argument("--time-scale", type=int, default=10)  # speedup factor
parser.add_argument("--random-seed", type=int, default=0)
parser.add_argument("--detailed-log", action="store_true", help="whether to print detailed logs")
parser.add_argument("--heatmap-center", type=float, nargs=3, default=[0, 0, 0])  # the center of the supply heatmap (x, z are the 2D location and y is the height)
parser.add_argument("--start-range", type=float, default=1)  # the range of the start location
parser.add_argument("--start_height", type=float, default=2)  # the height of the start location
parser.add_argument("--engine-dir", type=str, default="../unity3d")  # path to unity executable
parser.add_argument("--map-dir", type=str, default="../map_data")  # path to map files
parser.add_argument("--map-id", type=int, default=1)  # id of the map
parser.add_argument("--use-depth", action="store_true")  # whether to use depth map
parser.add_argument("--resume", action="store_true")  # whether to resume training from a checkpoint
parser.add_argument("--checkpoint-path", type=str, default="./agent_track2_ppo", help="dir to checkpoint files")
parser.add_argument("--replay-interval", type=int, default=1, help="episode interval to save replay")
parser.add_argument("--record", action="store_true", help="whether to record the game")
parser.add_argument("--replay-suffix", type=str, default="", help="suffix of the replay filename")
parser.add_argument("--inference", action="store_true", help="whether to run inference")
parser.add_argument("--game_config", default='mode3.json')

# training config
parser.add_argument("--num-workers", type=int, default=0)
parser.add_argument("--eval-interval", type=int, default=None)
parser.add_argument("--run", type=str, default="PPO", help="The RLlib-registered algorithm to use.")
parser.add_argument("--stop_episodes", type=int, default=100000000)
parser.add_argument("--stop-timesteps", type=int, default=100000000)
parser.add_argument("--stop-reward", type=float, default=999999)
parser.add_argument("--train-batch-size", type=int, default=400)



class SupplyBattleMultiAgentEnv(MultiAgentEnv):
    def __init__(self, config: EnvContext):
        super().__init__()

        self.seed(config["random_seed"])
        self.server_port = BASE_WORKER_PORT + config.worker_index
        print(f">>> New instance {self} on port: {self.server_port}")
        print(f"Worker Index: {config.worker_index}, VecEnv Index: {config.vector_index}")
        cur_path = os.path.abspath(os.path.dirname(__file__))
        time_stamp = f"{self.server_port}"
        self.log_path = os.path.expanduser("%s/%s" % (cur_path, time_stamp))
        self.writer = SummaryWriter(self.log_path,comment='metric')
        with open(self.log_path + "log.txt", "w") as f:
            f.write(
                f">>> {self.__class__}, server_port: {self.server_port} , worker_index: {config.worker_index}, log:\n")

        self.game = Game(
            map_dir=args.map_dir,
            engine_dir=args.engine_dir,
            server_port=self.server_port)
        self.game.set_game_config(config["game_config"])

        self.start_height = config["start_height"]
        env_config = config
        self.game.set_map_id(env_config["map_id"])
        self.game.set_episode_timeout(env_config["timeout"])
        self.game.set_random_seed(env_config["random_seed"])
        self.game.set_supply_heatmap_radius(60)
        self.game.set_supply_indoor_richness(10)  # 10
        self.game.set_supply_outdoor_richness(10)  # 10
        self.game.set_supply_indoor_quantity_range(10, 20)
        self.game.set_supply_outdoor_quantity_range(1, 5)
        self.game.set_supply_spacing(4)
        game_config = self.game.get_game_config()
        self.supply_heatmap_center = config['heatmap_center']
        self.args = config
        print(game_config)
        self.agent_ids = [agent["id"] for agent in game_config["agentSetups"]]
        for agent in game_config["agentSetups"]:
            self._agent_ids.add(agent["id"])
        if env_config["record"]:
            self.game.turn_on_record()
            self.game.set_game_replay_suffix(config["replay_suffix"])

        self.action_pools = {
            ActionVariable.WALK_DIR: [0, 90, 180, 270],
            ActionVariable.WALK_SPEED: [0, 5, 10],
            ActionVariable.TURN_LR_DELTA: [-2, 0, 2],
            ActionVariable.LOOK_UD_DELTA: [-1, 0, 1],
            ActionVariable.JUMP: [True, False],
            ActionVariable.ATTACK: [False, True],
            ActionVariable.RELOAD: [False, True],
            ActionVariable.PICKUP: [True],
        }
        self.action_space = spaces.MultiDiscrete([len(pool) for pool in self.action_pools.values()])
        self.game.set_available_actions([action_name for action_name in self.action_pools.keys()])

        list_spaces = [
                spaces.Box(-np.Inf, np.Inf, (7,), dtype=np.float32),
                spaces.Box(-np.Inf, np.Inf, (4,), dtype=np.float32),
                spaces.Box(-np.Inf, np.Inf, (3,), dtype=np.float32),
                spaces.Box(-np.Inf, np.Inf, (3,7), dtype=np.float32),
            ]
        if env_config["use_depth"]:
            self.game.turn_on_depth_map()
            width , height , max_depth = self.game.get_depth_map_size()
            list_spaces.append(spaces.Box(0, max_depth, (height, width), dtype=np.float32))
        self.observation_space = spaces.Tuple(list_spaces)


        self.episode_num = 0
        self.time_steps = 0

        self.game.init()

    def reset(self, record=False, replay_suffix=""):
        print("Reset for a new game ...")
        self.running_steps = 0

        self._reset_game_config()
        self.game.new_episode()

        obs_dict = {}
        self.state_dict = {}
        for agent_id in self.agent_ids:
            self.state_dict[agent_id] = self.game.get_state(agent_id)
            obs_dict[agent_id] = self._get_obs(self.state_dict[agent_id],agent_id)
        self.episode_num+=1
        self.hurt_num = 0
        self.dead_num = 0
        return obs_dict




    def step(self, action_dict):

        action_cmd_dict = {agent_id: self._action_process(action_dict[agent_id]) for agent_id in action_dict}
        self.game.make_action(action_cmd_dict)
        self.running_steps += 1
        self.time_steps += 1

        obs_dict = {}
        reward_dict = {}
        done_dict = {}
        info_dict = {}

        done_dict["__all__"] = self.game.is_episode_finished()

        for agent_id in action_dict:

            state = self.game.get_state(agent_id)
            if state.hit_enemy:
                self.hurt_num += 1
            if state.health <= 0:
                self.dead_num +=1

            self.state_dict[agent_id] = state
            reward_dict[agent_id] = self._compute_reward(state, agent_id)
            obs_dict[agent_id] = self._get_obs(state,agent_id)
            done_dict[agent_id] = self.game.is_episode_finished()
            info_dict[agent_id] = {}
            self.collected_supplys[agent_id] = state.num_supply
        if self.running_steps % 100 ==0:
            with open(self.log_path + "log.txt", "a") as f:
                f.write(f"{self.episode_num}-{self.running_steps} : {self.collected_supplys} \n")
        self.writer.add_scalar('hit_num',self.hurt_num,global_step=self.time_steps)
        self.writer.add_scalar('dead_num',self.hurt_num,global_step=self.time_steps)




        return obs_dict, reward_dict, done_dict, info_dict

    def close(self):
        self.game.close()
        self.writer.close()
        return super().close()

    def render(self, mode="replay"):
        return None

    def _reset_game_config(self):
        self.collected_supplys = {agent_id: 0 for agent_id in self.agent_ids}
        self.target_supply = {agent_id: None for agent_id in self.agent_ids}
        for agent_id in self.agent_ids:
            x = np.random.randint(-20, 20)
            y = self.start_height
            z = np.random.randint(-20, 20)
            self.game.set_start_location([x, y, z], agent_id)

    def _get_obs(self, state, agent_id):

        self.np_enemy_states = [
                [
                    enemy.position_x,
                    enemy.position_y,
                    enemy.position_z,
                    enemy.move_dir_x,
                    enemy.move_dir_y,
                    enemy.move_dir_z,
                    enemy.move_speed,

                ]
            for enemy in state.enemy_states]
        enemy_distance = [get_distance([enemy[0],enemy[1],enemy[2]], get_position(state)) for enemy in self.np_enemy_states]

        enemy_states = [[0 for _ in range(7)] for _ in range(3)]

        self.np_enemy_states.sort(key= lambda x:get_distance([x[0],x[1],x[2]], get_position(state)))
        for i in range(len(self.np_enemy_states)):
            if i>=3:
                break
            enemy_states[i] = self.np_enemy_states[i]


        self.np_supply_states = [
                np.asarray(
                    [
                        supply.position_x,
                        supply.position_y,
                        supply.position_z,
                    ]
                )
                for supply in state.supply_states
            ]

            # reinitialize: get target information

        supply_distances = [get_distance([supply[0], supply[1], supply[2]], get_position(state)) for supply in
                            self.np_supply_states]
        # target supply is the closest supply

        if self.target_supply[agent_id] is None:
            if len(supply_distances) >0:
                self.target_supply[agent_id] = self.np_supply_states[supply_distances.index(min(supply_distances))].tolist()
            else:
                self.target_supply[agent_id] = self.supply_heatmap_center
        target = self.target_supply[agent_id]

        x = state.position_x
        y = state.position_y
        z = state.position_z
        move_x = state.move_dir_x
        move_y = state.move_dir_y
        move_z = state.move_dir_z
        move_speed = state.move_speed
        pitch = state.pitch
        yaw = state.yaw

        #to do add
        weapon = [state.weapon_ammo,state.spare_ammo]



        obs = [[x,y,z,move_x,move_y,move_z,move_speed],[pitch,yaw,state.weapon_ammo,state.spare_ammo],target,enemy_states]
        if self.args["use_depth"]:
            obs.append(state.depth_map.tolist())
        if self.running_steps <=5:
            self.target_supply[agent_id]=None
        print(obs)
        return obs



    def _compute_reward(self, state, agent_id):
        '''
        奖励设计，（1）当前位置和目标位置的距离（2）装弹的负奖励（3）打到敌人正奖励（4）被敌人打到负奖励
        （5）捡到物资正奖励
        '''
        reward = -0.1
        if not self.game.is_episode_finished():
            # if self.target_supply[agent_id] is None:
            #     return reward
            # distance = get_distance([self.target_supply[agent_id][0], self.target_supply[agent_id][1],
            #                         self.target_supply[agent_id][2]],get_position(state))
            # reward += -distance
            # if get_distance([self.target_supply[agent_id][0], self.target_supply[agent_id][1],
            #                         self.target_supply[agent_id][2]],get_position(state))<=1:
            #     self.target_supply[agent_id]=None
            #     reward +=100
            #     reward += (state.num_supply - self.collected_supplys[agent_id])*10

            if state.hit_enemy:
                reward+=10
            if state.hit_by_enemy:
                reward-=10

        return reward




    def _action_process(self, action):
        action_values = list(self.action_pools.values())
        return [action_values[i][action[i]] for i in range(len(action))]



if __name__ == "__main__":
    import ray
    from ray.rllib.agents.ppo import PPOTrainer
    from ray.tune.logger import pretty_print
    # from envs.envs_track2 import SupplyGatherDiscreteSingleTarget

    args = parser.parse_args()
    # env = SupplyBattleMultiAgentEnv(vars(args))
    # state = env.reset()
    # print(state)
    # env.close()


    ray.init()
    agent = PPOTrainer(
        config={
            "env": SupplyBattleMultiAgentEnv,
            "env_config": vars(args),
            "framework": "torch",
            "num_workers": args.num_workers,
            "evaluation_interval": args.eval_interval,
            "train_batch_size": args.train_batch_size,  # default of ray is 4000
        }
    )
    step = 0
    while True:
        result = agent.train()
        print(pretty_print(result))
        if step !=0 and step %30==0:
            os.makedirs(args.checkpoint_dir, exist_ok=True)
            agent.save_checkpoint(args.checkpoint_dir)
        step+=1
        if result["episodes_total"] >= args.stop_episodes:
            # agent.save_checkpoint(args.checkpoint_path)
            os.makedirs(args.checkpoint_dir, exist_ok=True)
            agent.save_checkpoint(args.checkpoint_dir)
            agent.stop()
            break
    print('training is done-----------------over')
    ray.shutdown()
    sys.exit()
