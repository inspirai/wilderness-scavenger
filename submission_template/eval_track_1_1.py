from inspirai_fps import ActionVariable


# evaluation configs
USED_ACTIONS = [
    ActionVariable.WALK_DIR,
    ActionVariable.WALK_SPEED,
    ActionVariable.TURN_LR_DELTA,
    ActionVariable.LOOK_UD_DELTA,
    ActionVariable.JUMP,
]


def run_eval(args, eval_id=None):
    from common import (
        TURN_ON_RECORDING,
        DEPTH_MAP_WIDTH,
        DEPTH_MAP_HEIGHT,
        DEPTH_MAP_FAR,
        RunningStatus,
        DEFAULT_PAYLOAD,
        send_results
    )

    from inspirai_fps import Game
    from inspirai_fps.utils import get_position
    from submission.agents import AgentNavigation

    import random
    from functools import partial
    from rich.console import Console

    random.seed(args.seed)
    print = partial(Console().print, style="bold magenta")

    # configure game
    game = Game(map_dir=args.map_dir, engine_dir=args.engine_dir)
    game.set_game_mode(Game.MODE_NAVIGATION)
    game.set_random_seed(args.seed)
    game.set_available_actions(USED_ACTIONS)
    game.set_episode_timeout(args.episode_timeout)
    game.set_depth_map_size(DEPTH_MAP_WIDTH, DEPTH_MAP_HEIGHT, DEPTH_MAP_FAR)
    game.init()

    results = []
    ep_idx = 0
    
    data = DEFAULT_PAYLOAD.copy()
    data.update({
        "id": eval_id,
        "status": RunningStatus.STARTED,
        "current_episode": ep_idx,
        "total_episodes": len(args.map_list) * args.episodes_per_map
    })
    send_results(data)

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

            data.update({
                "current_episode": ep_idx,
                "average_time_use": sum(r["used_time"] for r in results) / len(results),
                "average_time_punish": sum(r["punish_time"] for r in results) / len(results),
                "success_rate": sum(r["reach_target"] for r in results) / len(results)
            })
            data["average_time_total"] = data["average_time_use"] + data["average_time_punish"]
            send_results(data)

    game.close()

    data.update({"status": RunningStatus.FINISHED})
    send_results(data)
    