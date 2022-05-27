import os
import random
from inspirai_fps import Game, ActionVariable

import cv2
import numpy as np
from PIL import Image
from functools import partial
from rich.console import Console

print = partial(Console().print, style="bold magenta")

SCALE = 1
FAR = 200
WIDTH = 500
HEIGHT = round(WIDTH / 16 * 9)


def visualize(depth_map, far=FAR, scale=SCALE):
    img = ((1 - depth_map / far) * 255).astype(np.uint8)
    h, w = [x * scale for x in img.shape]
    img = cv2.resize(img, (w, h))
    img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img)


class PolicyPool:
    """ a pool of policies -> static methods for easy access """

    @staticmethod
    def simple_rotate(state):
        return [
            (ActionVariable.TURN_LR_DELTA, 1),
        ]

    @staticmethod
    def simple_walk_forward(state):
        return [
            (ActionVariable.WALK_DIR, 0),
            (ActionVariable.WALK_SPEED, 5),
        ]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--map-dir", type=str, default="/mnt/d/Codes/cog-local/map-data")
    parser.add_argument("--engine-dir", type=str, default="/mnt/d/Codes/cog-local/fps_linux")
    parser.add_argument("--width", type=int, default=WIDTH)
    parser.add_argument("--height", type=int, default=HEIGHT)
    parser.add_argument("--far", type=int, default=FAR)
    parser.add_argument("--scale", type=int, default=SCALE)
    parser.add_argument("--map-id-list", type=int, nargs="+", default=[])
    parser.add_argument("--tag", type=str, default=None)
    parser.add_argument("--save-dir", type=str, default="VisDepth")
    parser.add_argument("--policy", type=str, default="simple_rotate")
    args = parser.parse_args()

    policy = getattr(PolicyPool, args.policy)
    os.makedirs(args.save_dir, exist_ok=True)
    map_dir = os.path.expanduser(args.map_dir)
    engine_dir = os.path.expanduser(args.engine_dir)

    game = Game(map_dir, engine_dir)
    game.set_available_actions(
        [
            ActionVariable.WALK_DIR,
            ActionVariable.WALK_SPEED,
            ActionVariable.TURN_LR_DELTA,
            ActionVariable.LOOK_UD_DELTA,
        ]
    )
    game.set_episode_timeout(10)
    game.turn_on_depth_map()
    game.set_depth_map_size(args.width, args.height, args.far)
    game.set_game_mode(Game.MODE_NAVIGATION)
    game.init()

    for map_id in args.map_id_list:
        game.set_map_id(map_id)
        game.set_start_location([0, 0, 0])
        game.turn_on_record()
        game.set_game_replay_suffix(f"vis_depth_{args.tag}")
        game.new_episode()

        frames = []
        while not game.is_episode_finished():
            state_all = game.get_state_all()
            action_all = {
                agent_id: policy(state)
                for agent_id, state in state_all.items()
            }

            game.make_action_by_list(action_all)

            print(state_all)
            print(action_all)

            frames.append(visualize(state_all[0].depth_map, args.far, args.scale))

        print(f"Map {map_id:03d} finished")

        save_name = f"policy[{args.policy}]_map[{map_id:03d}]_WxHxF={args.width}x{args.height}x{args.far}"
        save_path = os.path.join(args.save_dir, save_name)

        frames[0].save(
            save_path + ".gif",
            format="GIF",
            append_images=frames[1:],
            save_all=True,
            duration=1000 // 50 * 5,
            loop=0,
        )

    game.close()
