import time
import random
import argparse
import numpy as np

from inspirai_fps import Game, ActionVariable
from inspirai_fps.utils import get_position

from rich.progress import track
from rich.console import Console

console = Console()

parser = argparse.ArgumentParser()
parser.add_argument("--port", type=int, default=50051)
parser.add_argument("--timeout", type=int, default=10)
parser.add_argument("--map-id", type=int, default=1)
parser.add_argument("--random-seed", type=int, default=0)
parser.add_argument("--num-episodes", type=int, default=1)
parser.add_argument("--engine-dir", type=str, default="../unity3d")
parser.add_argument("--map-dir", type=str, default="../data")
parser.add_argument("--use-depth-map", action="store_true")
parser.add_argument("--record", action="store_true")
parser.add_argument("--replay-suffix", type=str, default="")
parser.add_argument("--start-location", type=float, nargs=3, default=[0, 0, 0])
parser.add_argument("--target-location", type=float, nargs=3, default=[5, 0, 5])
parser.add_argument("--walk-speed", type=float, default=1)
args = parser.parse_args()


def get_pitch_yaw(x, y, z):
    pitch = np.arctan2(y, (x**2 + z**2) ** 0.5) / np.pi * 180
    yaw = np.arctan2(x, z) / np.pi * 180
    return pitch, yaw


def my_policy(state):
    """Define a simple navigation policy"""
    self_pos = [state.position_x, state.position_y, state.position_z]
    target_pos = args.target_location
    direction = [v2 - v1 for v1, v2 in zip(self_pos, target_pos)]
    yaw = get_pitch_yaw(*direction)[1]
    action = [yaw, args.walk_speed]
    return action


# valid actions
used_actions = [
    ActionVariable.WALK_DIR,
    ActionVariable.WALK_SPEED,
]

# instantiate Game
game = Game(map_dir=args.map_dir, engine_dir=args.engine_dir)
game.set_game_mode(Game.MODE_NAVIGATION)
game.set_episode_timeout(args.timeout)
game.set_start_location(args.start_location)  # set start location of the first agent
game.set_target_location(args.target_location)
game.set_available_actions(used_actions)
game.set_map_id(args.map_id)

if args.use_depth_map:
    game.turn_on_depth_map()

if args.record:
    game.turn_on_record()

game.init()

for ep in track(range(args.num_episodes), description="Running Episodes ..."):
    game.set_game_replay_suffix(f"{args.replay_suffix}_episode_{ep}")
    game.new_episode()

    while not game.is_episode_finished():
        ts = game.get_time_step()

        t = time.perf_counter()
        state_all = game.get_state_all()
        action_all = {
            agent_id: my_policy(state_all[agent_id]) for agent_id in state_all
        }
        game.make_action(action_all)
        dt = time.perf_counter() - t

        for agent_id, state in state_all.items():
            step_info = {
                "Episode": ep,
                "TimeStep": ts,
                "AgentID": agent_id,
                "Location": get_position(state),
                "Action": {
                    name: val for name, val in zip(used_actions, action_all[agent_id])
                },
                "#SupplyInfo": len(state.supply_states),
                "#EnemyInfo": len(state.enemy_states),
                "StepRate": round(1 / dt),
            }
            if args.use_depth_map:
                step_info["DepthMap"] = state.depth_map.shape
            console.print(step_info, style="bold magenta")

    print("episode ended ...")

game.close()
