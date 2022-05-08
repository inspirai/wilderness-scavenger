import os
import random
from rich.console import Console
from inspirai_fps import Game, ActionVariable

import cv2
import numpy as np

SCALE = 1
WIDTH = 380
HEIGHT = 220
FAR = 100

console = Console()


def visualize(map_id, ts, depth_map):
    img = (depth_map / FAR * 255).astype(np.uint8)
    h, w = img.shape
    img = cv2.resize(img, (w * SCALE, h * SCALE))
    img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
    cv2.imwrite(f"dmp/depth_map={map_id}_ts={ts}.png", img)


os.makedirs("dmp", exist_ok=True)
map_dir = os.path.expanduser("~/map_data")
engine_dir = os.path.expanduser("~/newGame_linux")

game = Game(map_dir, engine_dir)
game.set_available_actions(
    [
        ActionVariable.WALK_DIR,
        ActionVariable.WALK_SPEED,
    ]
)
game.set_episode_timeout(10)
game.turn_on_depth_map()
game.set_depth_map_size(WIDTH, HEIGHT, FAR)
game.set_game_mode(Game.MODE_NAVIGATION)
# game.add_agent(agent_name="agent_1")


game.init()
for ep in range(5):
    map_id = random.randint(1, 100)
    game.set_map_id(map_id)
    game.random_start_location(indoor=random.choice([True, False]))
    console.print(game.get_game_config(), style="bold magenta")

    game.new_episode()

    while not game.is_episode_finished():
        state_all = game.get_state_all()
        action_all = {
            agent_id: [
                random.choice([0, 90, 180, 270]),
                random.choice([0, 5, 10]),
            ]
            for agent_id in state_all
        }

        game.make_action(action_all)

        console.print(state_all, style="bold magenta")
        console.print(action_all, style="bold magenta")

        depth_map = state_all[0].depth_map

        visualize(ep + 1, game.get_time_step(), depth_map)

    console.print("Map {} finished".format(map_id), style="bold magenta")

game.close()
