from inspirai_fps import ActionVariable

# evaluation configs
NUM_AGENTS = 10
USED_ACTIONS = [
    ActionVariable.WALK_DIR,
    ActionVariable.WALK_SPEED,
    ActionVariable.TURN_LR_DELTA,
    ActionVariable.LOOK_UD_DELTA,
    ActionVariable.JUMP,
    ActionVariable.PICKUP,
    ActionVariable.ATTACK,
    ActionVariable.RELOAD,
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
    "supply_radius": [100],
    "supply_richness_outdoor": [10, 20, 30],
    "supply_richness_indoor": [70, 80, 90],
    "supply_spacing": [5],
    "supply_indoor_quantity_range": {
        "qmin": 10,
        "qmax": 50,
    },
    "supply_outdoor_quantity_range": {
        "qmin": 1,
        "qmax": 5,
    },
    "supply_refresh": {
        "refresh_time": [600],
        "heatmap_radius": [30, 50, 70],
        "outdoor_richness": [10, 20, 30],
        "indoor_richness": [70, 80, 90],
    },
}


def run_eval(game, args, message_data):
    from inspirai_fps import Game
    from inspirai_fps.utils import get_position
    from submission.agents import AgentSupplyBattle

    import random
    from functools import partial
    from rich.console import Console
    from common import (
        DEPTH_MAP_WIDTH,
        DEPTH_MAP_HEIGHT,
        DEPTH_MAP_FAR,
        RunningStatus,
        send_results,
    )

    random.seed(args.seed)
    print = partial(Console().print, style="bold magenta")

    game.set_random_seed(args.seed)
    game.set_game_mode(Game.MODE_SUP_BATTLE)
    game.set_episode_timeout(args.episode_timeout)
    game.set_available_actions(USED_ACTIONS)
    game.set_depth_map_size(DEPTH_MAP_WIDTH, DEPTH_MAP_HEIGHT, DEPTH_MAP_FAR)
    for agent_id in range(1, NUM_AGENTS):
        game.add_agent()

    results = []
    ep_idx = 0

    message_data.update({"current_episode": ep_idx})
    send_results(message_data)

    for map_id in args.map_list:
        game.set_map_id(map_id)

        for ep in range(args.episodes_per_map):
            for agent_id in range(NUM_AGENTS):
                game.random_start_location(agent_id)

            game.set_supply_heatmap_center(
                random.choice(SUPPLY_CONFIGS["supply_center"])
            )
            game.set_supply_heatmap_radius(
                random.choice(SUPPLY_CONFIGS["supply_radius"])
            )
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

            game.clear_supply_refresh()

            refresh_config_pool = SUPPLY_CONFIGS["supply_refresh"]
            heatmap_centers = []
            heatmap_radius = []

            for time in refresh_config_pool["refresh_time"]:
                center = random.choice(game.get_valid_locations()["indoor"])
                radius = random.choice(refresh_config_pool["heatmap_radius"])

                game.add_supply_refresh(
                    refresh_time=time,
                    heatmap_center=center,
                    heatmap_radius=radius,
                    indoor_richness=random.choice(
                        refresh_config_pool["indoor_richness"]
                    ),
                    outdoor_richness=random.choice(
                        refresh_config_pool["outdoor_richness"]
                    ),
                )

                heatmap_centers.append(center)
                heatmap_radius.append(radius)

            print(game.get_game_config())

            game.new_episode()

            episode_info = {
                "start_location": game.get_start_location(),
                "supply_heatmap_center": game.get_supply_heatmap_center(),
                "supply_heatmap_radius": game.get_supply_heatmap_radius(),
                "refresh_time": refresh_config_pool["refresh_time"],
                "refresh_heatmap_center": heatmap_centers,
                "refresh_heatmap_radius": heatmap_radius,
                "time_step_per_action": game.time_step_per_action,
            }

            agents = {}
            for agent_id in range(NUM_AGENTS):
                if agent_id == 0:
                    agents[agent_id] = AgentSupplyBattle(episode_info)  # Your agent here
                else:
                    info = episode_info.copy()
                    info["start_location"] = game.get_start_location(agent_id)
                    agents[agent_id] = AgentSupplyBattle(info)  # will be replaced with our robot agent

            print(f">>>>>> Map {map_id:03d} - Episode {ep} <<<<<<")

            while not game.is_episode_finished():
                ts = game.get_time_step()
                state_all = game.get_state_all()
                action_all = {
                    agent_id: agents[agent_id].act(ts, state_all[agent_id])
                    for agent_id in range(NUM_AGENTS)
                }
                game.make_action(action_all)

                if ts % 50 == 0:
                    walk_dir = round(action_all[0].walk_dir, 2)
                    curr_loc = [round(x, 2) for x in get_position(state_all[0])]
                    num_supply = state_all[0].num_supply
                    print(
                        f"{map_id=}\t{ep=}\t{ts=}\t{curr_loc=}\t{num_supply=}\t{walk_dir=}"
                    )

            res = game.get_game_result()
            results.append(res)

            print(f">>>>> Episode ends <<<<<")
            print(res)

            ep_idx += 1
            message_data.update(
                {
                    "current_episode": ep_idx,
                    "average_supply": sum(r["num_supply"] for r in results)
                    / len(results),
                }
            )
            send_results(message_data)

    message_data.update({"status": RunningStatus.FINISHED})
    send_results(message_data)
