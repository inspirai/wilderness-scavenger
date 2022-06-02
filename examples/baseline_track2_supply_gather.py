import argparse

parser = argparse.ArgumentParser()

# game setup
parser.add_argument("--timeout", type=int, default=60 * 2)  # The time length of one game (sec)
parser.add_argument("--time-scale", type=int, default=10)  # speedup factor
parser.add_argument("--random-seed", type=int, default=0)
parser.add_argument("--detailed-log", action="store_true", help="whether to print detailed logs")
parser.add_argument("--heatmap-center", type=float, nargs=3, default=[8, 8])  # the center of the supply heatmap (x, z are the 2D location and y is the height)
parser.add_argument("--start-range", type=float, default=1)  # the range of the start location
parser.add_argument("--start-hight", type=float, default=5)  # the height of the start location
parser.add_argument("--engine-dir", type=str, default="../wildscav-linux-backend")  # path to unity executable
parser.add_argument("--map-dir", type=str, default="../map_data")  # path to map files
parser.add_argument("--map-id", type=int, default=1)  # id of the map
parser.add_argument("--use-depth-map", action="store_true")  # whether to use depth map
parser.add_argument("--resume", action="store_true")  # whether to resume training from a checkpoint
parser.add_argument("--checkpoint-dir", type=str, default="checkpoints_track2", help="dir to checkpoint files")
parser.add_argument("--replay-interval", type=int, default=1, help="episode interval to save replay")
parser.add_argument("--record", action="store_true", help="whether to record the game")
parser.add_argument("--replay-suffix", type=str, default="", help="suffix of the replay filename")
parser.add_argument("--inference", action="store_true", help="whether to run inference")

# training config
parser.add_argument("--num-workers", type=int, default=0)
parser.add_argument("--eval-interval", type=int, default=None)
parser.add_argument("--run", type=str, default="PPO", help="The RLlib-registered algorithm to use.")
parser.add_argument("--stop-iters", type=int, default=9999)
parser.add_argument("--stop-timesteps", type=int, default=100000000)
parser.add_argument("--stop-reward", type=float, default=999999)
parser.add_argument("--stop-episodes", type=float, default=100000)
parser.add_argument("--train-batch-size", type=int, default=800)


if __name__ == "__main__":
    import os
    import ray
    from ray.rllib.agents.ppo import PPOTrainer
    from ray.tune.logger import pretty_print
    from envs.envs_track2 import SupplyGatherDiscreteSingleTarget

    args = parser.parse_args()

    ray.init()
    trainer = PPOTrainer(
        config={
            "env": SupplyGatherDiscreteSingleTarget,
            "env_config": vars(args),
            "framework": "torch",
            "num_workers": args.num_workers,
            "evaluation_interval": args.eval_interval,
            "train_batch_size": args.train_batch_size,  # default of ray is 4000
        }
    )

    if args.resume:
        trainer.load_checkpoint(args.checkpoint_dir)

    while True:
        result = trainer.train()
        print(pretty_print(result))
        if result["episodes_total"] >= args.stop_episodes:
            os.makedirs(args.checkpoint_dir, exist_ok=True)
            trainer.save_checkpoint(args.checkpoint_dir)
            trainer.stop()
            break

    print(pretty_print(result))
    ray.shutdown()
