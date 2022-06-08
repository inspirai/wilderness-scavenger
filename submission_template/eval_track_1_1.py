from inspirai_fps import ActionVariable


# evaluation configs
USED_ACTIONS = [
    ActionVariable.WALK_DIR,
    ActionVariable.WALK_SPEED,
    ActionVariable.TURN_LR_DELTA,
    ActionVariable.LOOK_UD_DELTA,
    ActionVariable.JUMP,
]


def run_eval(game, args, message_data):
    from inspirai_fps import Game
    from inspirai_fps.utils import get_position
    from submission.agents import AgentNavigation

    import random
    from functools import partial
    from rich.console import Console
    from common import (
        DEPTH_MAP_WIDTH,
        DEPTH_MAP_HEIGHT,
        DEPTH_MAP_FAR,
        RunningStatus,
        send_results
    )

    random.seed(args.seed)
    print = partial(Console().print, style="bold magenta")

    # configure game
    game.set_game_mode(Game.MODE_NAVIGATION)
    game.set_random_seed(args.seed)
    game.set_available_actions(USED_ACTIONS)
    game.set_episode_timeout(args.episode_timeout)
    game.set_depth_map_size(DEPTH_MAP_WIDTH, DEPTH_MAP_HEIGHT, DEPTH_MAP_FAR)

    results = []
    ep_idx = 0
    
    message_data.update({"current_episode": ep_idx})
    send_results(message_data)

    for map_id in args.map_list:
        game.set_map_id(map_id)

        for ep in range(args.episodes_per_map):
            game.random_start_location()
            game.random_target_location()
            game.new_episode()

            episode_info = {
                "start_location": game.get_start_location(),
                "target_location": game.get_target_location(),
                "time_step_per_action": game.time_step_per_action,
            }

            agent = AgentNavigation(episode_info)

            print(f">>>>>> Map {map_id:03d} - Episode {ep} <<<<<<")

            start = [round(x, 2) for x in episode_info["start_location"]]
            target = [round(x, 2) for x in episode_info["target_location"]]
            
            while not game.is_episode_finished():
                ts = game.get_time_step()
                state = game.get_state()
                action = agent.act(ts, state)
                game.make_action({0: action})

                if ts % game.frame_rate == 0:
                    curr_location = [round(x, 2) for x in get_position(state)]
                    print(
                        f"{map_id=}\t{ep=}\t{ts=}\t{start=} => {target=}\t{curr_location=}"
                    )

            res = game.get_game_result()
            results.append(res)

            print(f">>>>> Episode ends <<<<<")
            print(res)

            ep_idx += 1

            message_data.update({
                "current_episode": ep_idx,
                "average_time_use": sum(r["used_time"] for r in results) / len(results),
                "average_time_punish": sum(r["punish_time"] for r in results) / len(results),
                "success_rate": sum(r["reach_target"] for r in results) / len(results)
            })
            message_data["average_time_total"] = message_data["average_time_use"] + message_data["average_time_punish"]
            send_results(message_data)

    message_data.update({"status": RunningStatus.FINISHED})
    send_results(message_data)
    