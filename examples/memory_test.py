import os
import random
from inspirai_fps import Game, ActionVariable


MAPDIR = os.path.expanduser("~/map_data")
ENGINEDIR = os.path.expanduser("~/fps_linux")


def rollout(
    env_id,
):
    game = Game(server_port=50000 + env_id, map_dir=MAPDIR, engine_dir=ENGINEDIR)
    game.set_available_actions(
        [
            ActionVariable.WALK_DIR,
            ActionVariable.WALK_SPEED,
        ]
    )
    game.set_game_mode(Game.MODE_SUP_BATTLE)
    game.turn_on_depth_map()
    game.init()

    done = True
    while True:
        if done:
            game.set_map_id(random.randint(1, 3))
            game.set_episode_timeout(30)
            game.set_target_location(game.get_start_location())
            game.random_start_location()
            game.turn_on_record()
            game.set_game_replay_suffix("temp")
            game.new_episode()
            print(f"new episode in process {env_id}")

        game.make_action({0: [0, 9]})
        _ = game.get_state_all()
        done = game.is_episode_finished()


if __name__ == "__main__":
    from multiprocessing import Process

    for i in range(5):
        Process(target=rollout, args=(i,)).start()
