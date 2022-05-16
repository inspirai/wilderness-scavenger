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
WIDTH = 500
HEIGHT = round(WIDTH / 16 * 9)
FAR = 200
NUM_EPISODES = 1


def visualize(depth_map):
    img = ((1 - depth_map / FAR) * 255).astype(np.uint8)
    h, w = img.shape
    img = cv2.resize(img, (w * SCALE, h * SCALE))
    img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img)


os.makedirs("dmp", exist_ok=True)
map_dir = os.path.expanduser("~/map-data")
engine_dir = os.path.expanduser("~/fps_linux")

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
game.set_depth_map_size(WIDTH, HEIGHT, FAR)
game.set_game_mode(Game.MODE_NAVIGATION)
game.init()

tag = "rotate"

for ep in range(NUM_EPISODES):
    map_id = random.randint(1, 100)
    game.set_map_id(map_id)
    game.set_start_location([0, 0, 0])
    game.turn_on_record()
    game.set_game_replay_suffix("viz_depth_map")
    game.new_episode()

    frames = []
    while not game.is_episode_finished():
        state_all = game.get_state_all()
        action_all = {
            agent_id: [
                0,  # walk_dir
                -2,  # walk_speed
                1,  # turn_lr_delta
                0,  # look_ud_delta
            ]
            for agent_id in state_all
        }

        game.make_action(action_all)

        print(state_all)
        print(action_all)

        frames.append(visualize(state_all[0].depth_map))

    print(f"Map {map_id:03d} finished")

    frames[0].save(
        f"dmp/{map_id=}_{tag}.gif",
        format="GIF",
        append_images=frames[1:],
        save_all=True,
        duration=1000 // 50 * 5,
        loop=0,
    )

game.close()
