from re import S
import time
import random
import argparse
from turtle import st

from rich.progress import track
from rich.console import Console

console = Console()
import numpy as np
from inspirai_fps import Game, ActionVariable
from inspirai_fps.utils import get_position


parser = argparse.ArgumentParser()
parser.add_argument("--port", type=int, default=50051)
parser.add_argument("--timeout", type=int, default=10)
parser.add_argument("--map-id", type=int, default=1)
parser.add_argument("--random-seed", type=int, default=0)
parser.add_argument("--num-episodes", type=int, default=1)
parser.add_argument("--engine-dir", type=str, default="../wildscav-linux-backend")
parser.add_argument("--map-dir", type=str, default="../map_data")
parser.add_argument("--num-agents", type=int, default=10)
parser.add_argument("--use-depth-map", action="store_true")
parser.add_argument("--record", action="store_true")
parser.add_argument("--replay-suffix", type=str, default="")
parser.add_argument("--start-location", type=float, nargs=3, default=[0, 0, 0])
parser.add_argument("--walk-speed", type=float, default=5)
args = parser.parse_args()

def get_picth_yaw(x, y, z):
    pitch = np.arctan2(y, (x**2 + z**2)**0.5) / np.pi * 180
    yaw = np.arctan2(x, z) / np.pi * 180

    return pitch, yaw

def act(state,last_enemy):
    x,y,z = state.position_x,state.position_y,state.position_z
    l_x,l_y,l_z = last_enemy.position_x,last_enemy.position_y,last_enemy.position_z
    x_,y_,z_ = l_x-x,l_y-y,l_z-z
    pitch_,yaw_ = get_picth_yaw(x_,y_,z_)
    cur_pitch,cur_yaw = state.pitch,state.yaw
    pitch=pitch_ + cur_pitch
    yaw = yaw_ - cur_yaw
    # if yaw<-180:
    #     yaw+=360
    # elif yaw>180:
    #     yaw-=360
    if yaw>=180:
        yaw-=360
    elif yaw<=-180:
        yaw+=360

    if yaw_<0:
        yaw_+=360
    yaw/=5
    pitch/=5
    if yaw>1:
        yaw=1
    elif yaw<-1:
        yaw=-1
    if pitch>1:
        pitch=1
    elif pitch<-1:
        pitch=-1
    return [yaw_,pitch,yaw]




# Define a random policy
def stay_policy(state):

    return [
        0,  # walk_dir
        0,  # walk_speed
        0,  # turn left right
        0,  # look up down
        False,
        False,  # attack
        False,  # reload
        True,  # collect
    ]

def random_policy(state):
    return [
        state.yaw,
        0,
        # random.randint(1, 10),  # walk_speed
        # random.choice([-1, 0, 1]),  # turn_lr_delta
        1,
        0,  # turn_ud_delta
        0,  # jump
        False,
        False,
        True,
    ]
#Define a rule fight policy
def rule_policy(state,last_enemy):
    if len(state.enemy_states)==0:
        last_enemy = {}
        return last_enemy ,random_policy(state)
    else:
        if last_enemy and last_enemy.id in state.enemy_states and last_enemy.health>0:
            walk_dir,pitch,yaw = act(state,last_enemy)
            last_enemy = state.enemy_states[last_enemy.id]
        else:
            if state.enemy_states:
                for id,info in state.enemy_states.items():
                    if info.health >0:
                        last_enemy = info
                        break
            if not last_enemy or last_enemy.health<=0:
                last_enemy={}
                return last_enemy ,random_policy(state)
            else:
                walk_dir,pitch,yaw = act(state,last_enemy)
        return last_enemy,[
            walk_dir,
            args.walk_speed,
            yaw,
            pitch,
            False,
            True if abs(yaw)<0.1 else False,
            False if state.weapon_ammo>0 else True,
            True]


# valid actions
used_actions = [
    ActionVariable.WALK_DIR,
    ActionVariable.WALK_SPEED,
    ActionVariable.TURN_LR_DELTA,
    ActionVariable.LOOK_UD_DELTA,
    ActionVariable.JUMP,
    ActionVariable.ATTACK,
    ActionVariable.RELOAD,
    ActionVariable.PICKUP,
]

# instantiate Game
game = Game(map_dir=args.map_dir, engine_dir=args.engine_dir)
game.set_game_mode(Game.MODE_SUP_BATTLE)
game.set_supply_heatmap_center([args.start_location[0], args.start_location[2]])
game.set_supply_heatmap_radius(50)
game.set_supply_indoor_richness(80)
game.set_supply_outdoor_richness(20)
game.set_supply_indoor_quantity_range(10, 50)
game.set_supply_outdoor_quantity_range(1, 5)
game.set_supply_spacing(5)
game.set_episode_timeout(args.timeout)
game.set_start_location(args.start_location)  # set start location of the first agent
game.set_available_actions(used_actions)
game.set_map_id(args.map_id)

if args.use_depth_map:
    game.turn_on_depth_map()

if args.record:
    game.turn_on_record()


game.add_agent()
game.set_start_location([0, 0, 5], 1)

game.add_agent()
game.set_start_location([5, 0, 0], 2)
game.add_agent()
game.set_start_location([0, 0, -5], 3)
game.add_agent()
game.set_start_location([-5, 0, 0], 4)

game.init()

for ep in track(range(args.num_episodes), description="Running Episodes ..."):
    game.set_game_replay_suffix(f"{args.replay_suffix}_episode_{ep}")
    game.new_episode()
    last_enemy = {}
    console.print(game.get_game_config())
    while not game.is_episode_finished():


        t = time.perf_counter()
        state_all = game.get_state_all()
        action_all = {
            agent_id: stay_policy(state_all[agent_id]) for agent_id in state_all
        }
        last_enemy, action_all[0] = rule_policy(state_all[0],last_enemy)
        # action_all[0] = random_policy(state_all[0])
        game.make_action(action_all)
        dt = time.perf_counter() - t



        agent_id = 0
        state = state_all[agent_id]
        step_info = {
            "time": game.get_time_step(),
            "AgentID": agent_id,
            "Location": get_position(state),
            "pitch":state.pitch,
            "yaw":state.yaw,
            "ammo":[state.weapon_ammo,state.spare_ammo],
            "walk_dir":[state.move_dir_x,state.move_dir_y,state.move_dir_z],
            "Action": {
                name: val for name, val in zip(used_actions, action_all[agent_id])
            },
            "#EnemyInfo": state.enemy_states,
            "last_enemy":last_enemy,

        }
        if args.use_depth_map:
            step_info["DepthMap"] = state.depth_map.shape

        if last_enemy:
            x,y,z = state.position_x,state.position_y,state.position_z
            l_x,l_y,l_z = last_enemy.position_x,last_enemy.position_y,last_enemy.position_z
            x_,y_,z_ = l_x-x,l_y-y,l_z-z
            pitch_,yaw = get_picth_yaw(x_,y_,z_)
            step_info["related_yaw"]= yaw

        console.print(step_info, style="bold magenta")

    print("episode ended ...")

game.close()