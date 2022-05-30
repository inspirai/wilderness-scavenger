from common import DEPTH_MAP_WIDTH, DEPTH_MAP_HEIGHT, DEPTH_MAP_FAR, TURN_ON_RECORDING
from common import RunningStatus, send_results
from inspirai_fps import ActionVariable


# evaluation configs
EPISODE_TIMEOUT = 60 * 10  # fixed to 10 minutes for online evaluation
USED_ACTIONS = [
    ActionVariable.WALK_DIR,
    ActionVariable.WALK_SPEED,
    ActionVariable.TURN_LR_DELTA,
    ActionVariable.LOOK_UD_DELTA,
    ActionVariable.JUMP,
    ActionVariable.PICKUP,
]
SUPPLY_CONFIGS = {
    "supply_center": [
        [0.0, 0.0],
        [0.0, 20],
        [0.0, -20],
        [20, 0.0],
        [20, 20],
        [20, -20],
        [-20, 0.0],
        [-20, 20],
        [-20, -20],
    ],
    "supply_radius": [
        10,
        30,
        50,
        70,
    ],
    "supply_richness_outdoor": [10, 20, 30],
    "supply_richness_indoor": [50, 70, 90],
    "supply_spacing": [
        5,
        10,
        15,
        20,
    ],
    "supply_indoor_quantity_range": {
        "qmin": 10,
        "qmax": 50,
    },
    "supply_outdoor_quantity_range": {
        "qmin": 1,
        "qmax": 5,
    },
}


def run_eval(args, eval_id=None):
    from inspirai_fps import Game
    from inspirai_fps.utils import get_position
    from submission.agents import AgentSupplyGathering

    import random
    from functools import partial
    from rich.console import Console

    random.seed(args.seed)
    print = partial(Console().print, style="bold magenta")

    game = Game(map_dir=args.map_dir, engine_dir=args.engine_dir)
    game.set_random_seed(args.seed)
    game.set_game_mode(Game.MODE_SUP_GATHER)
    game.set_episode_timeout(args.episode_timeout or EPISODE_TIMEOUT)
    game.set_available_actions(USED_ACTIONS)
    game.set_depth_map_size(DEPTH_MAP_WIDTH, DEPTH_MAP_HEIGHT, DEPTH_MAP_FAR)
    game.set_supply_heatmap_center(random.choice(SUPPLY_CONFIGS["supply_center"]))
    game.set_supply_heatmap_radius(random.choice(SUPPLY_CONFIGS["supply_radius"]))
    game.set_supply_outdoor_richness(
        random.choice(SUPPLY_CONFIGS["supply_richness_outdoor"])
    )
    game.set_supply_indoor_richness(
        random.choice(SUPPLY_CONFIGS["supply_richness_indoor"])
    )
    game.set_supply_spacing(random.choice(SUPPLY_CONFIGS["supply_spacing"]))
    game.set_supply_indoor_quantity_range(
        **SUPPLY_CONFIGS["supply_indoor_quantity_range"]
    )
    game.set_supply_outdoor_quantity_range(
        **SUPPLY_CONFIGS["supply_outdoor_quantity_range"]
    )
    game.init()

    results = []

    send_results(data={"status": "1"})

    ep_idx = 0
    for map_id in args.map_list:
        game.set_map_id(map_id)

        for ep in range(args.episodes_per_map):
            game.random_start_location()
            game.new_episode()

            episode_info = {
                "start_location": game.get_start_location(),
                "supply_heatmap_center": game.get_supply_heatmap_center(),
                "supply_heatmap_radius": game.get_supply_heatmap_radius(),
                "time_step_per_action": game.time_step_per_action,
            }

            agent = AgentSupplyGathering(episode_info)

            print(f">>>>>> Map {map_id:03d} - Episode {ep} <<<<<<")

            while not game.is_episode_finished():
                ts = game.get_time_step()
                state = game.get_state()
                action = agent.act(ts, state)
                game.make_action({0: action})

                if ts % 50 == 0:
                    walk_dir = round(action.walk_dir, 2)
                    curr_loc = [round(x, 2) for x in get_position(state)]
                    num_supply = state.num_supply
                    print(
                        f"{map_id=}\t{ep=}\t{ts=}\t{curr_loc=}\t{num_supply=}\t{walk_dir=}"
                    )

            res = game.get_game_result()
            results.append(res)

            print(f">>>>> Episode ends <<<<<")
            print(res)

            ep_idx += 1

            data = {
                "id": eval_id,
                "status": RunningStatus.STARTED,
                "current_episode": ep_idx,
                "total_episodes": len(args.map_list) * args.episodes_per_map,
                "num_supply": sum(r["num_supply"] for r in results) / len(results)
            }

            message = send_results(data)
            print("Response:", message)

    game.close()
    