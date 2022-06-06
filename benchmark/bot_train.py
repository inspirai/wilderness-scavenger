import argparse

parser = argparse.ArgumentParser()

# game setup
parser.add_argument("--timeout", type=int, default=60 * 2)  # The time length of one game (sec)
parser.add_argument("--time-scale", type=int, default=10)  # speedup factor
parser.add_argument("--random-seed", type=int, default=0)
parser.add_argument("--detailed-log", action="store_true", help="whether to print detailed logs")
parser.add_argument("--heatmap-center", type=float, nargs=3, default=[0,0,0])  # the center of the supply heatmap (x, z are the 2D location and y is the height)
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
parser.add_argument("--run", type=str, default="ppo", help="The RLlib-registered algorithm to use.")
parser.add_argument("--stop-iters", type=int, default=9999)
parser.add_argument("--stop-timesteps", type=int, default=100000000)
parser.add_argument("--stop-reward", type=float, default=999999)
parser.add_argument("--stop-episodes", type=float, default=100000)
parser.add_argument("--train-batch-size", type=int, default=4000)


if __name__ == "__main__":
    import os
    import ray
    from ray.rllib.agents.ppo import PPOTrainer
    from ray.rllib.agents.a3c import A3CTrainer
    from ray.rllib.agents.impala import ImpalaTrainer
    from ray.tune.logger import pretty_print
    from track2_env import SupplyGatherDiscreteSingleTarget

    args = parser.parse_args()
    
    alg = args.run

    ray.init()
    if alg =="ppo":
        trainer = PPOTrainer(
            config={
                "env": SupplyGatherDiscreteSingleTarget,
                "env_config": vars(args),
                "framework": "torch",
                "num_workers": args.num_workers,
                "evaluation_interval": args.eval_interval,
                "train_batch_size": args.train_batch_size,  # default of ray is 4000
                "ignore_worker_failures":True,
            }
        )
    elif alg =="a3c":
        trainer = A3CTrainer(
            config={
                "env": SupplyGatherDiscreteSingleTarget,
                "env_config": vars(args),
                "framework": "torch",
                "num_workers": args.num_workers,
                "evaluation_interval": args.eval_interval,
                "train_batch_size": args.train_batch_size,  # default of ray is 4000
                "ignore_worker_failures":True,
            }
        )
    elif alg =="impala":
        trainer = ImpalaTrainer(
            config={
                "env": SupplyGatherDiscreteSingleTarget,
                "env_config": vars(args),
                "framework": "torch",
                "num_workers": args.num_workers,
                "evaluation_interval": args.eval_interval,
                "train_batch_size": args.train_batch_size,  # default of ray is 4000
                "ignore_worker_failures":True,
                "num_gpus":0,
            }
        )

    if args.resume:
        trainer.restore(args.checkpoint_dir)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    step=1
    reward_max= 0
    while True:
        step += 1
        result = trainer.train()
        reward = result["episode_reward_mean"]
        
        if reward>reward_max:
            reward_max = reward
            trainer.save_(args.checkpoint_dir)
        print(f"current_train_steps:{step},episodes_total:{e},current_reward:{reward}")
        if result["episodes_total"] >= args.stop_episodes:
            trainer.save_(args.checkpoint_dir)
            trainer.stop()
            break

    print(pretty_print(result))
    ray.shutdown()
