import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-T", "--timeout", type=int, default=60 * 2)  # The time length of one game (sec)
parser.add_argument("-R", "--time-scale", type=int, default=10)
parser.add_argument("-M", "--map-id", type=int, default=1)
parser.add_argument("-S", "--random-seed", type=int, default=0)
parser.add_argument("--target-location", type=float, nargs=3, default=[0, 0, 0])
parser.add_argument("--start-location", type=float, nargs=3, default=[0, 0, 0])
parser.add_argument("--start-range", type=float, default=2)
parser.add_argument("--start-hight", type=float, default=5)
parser.add_argument("--engine-dir", type=str, default="../unity3d")
parser.add_argument("--map-dir", type=str, default="../data")
parser.add_argument("--num-workers", type=int, default=0)
parser.add_argument("--eval-interval", type=int, default=None)
parser.add_argument("--record", action="store_true")
parser.add_argument("--replay-suffix", type=str, default="")
parser.add_argument("--checkpoint-dir", type=str, default="checkpoints_track1")
parser.add_argument("--detailed-log", action="store_true", help="whether to print detailed logs")
parser.add_argument("--run", type=str, default="PPO", help="The RLlib-registered algorithm to use.")
parser.add_argument("--stop-iters", type=int, default=9999)
parser.add_argument("--stop-timesteps", type=int, default=100000000)
parser.add_argument("--stop-reward", type=float, default=95)


if __name__ == "__main__":
    import os
    import ray
    from ray.tune.logger import pretty_print
    from ray.rllib.agents.ppo import PPOTrainer
    from envs.envs_track1 import NavigationEnvSimple

    args = parser.parse_args()

    ray.init()
    trainer = PPOTrainer(
        config={
            "env": NavigationEnvSimple,
            "env_config": vars(args),
            "framework": "torch",
            "num_workers": args.num_workers,
            "evaluation_interval": args.eval_interval,
        }
    )

    while True:
        result = trainer.train()
        print(pretty_print(result))

        if result["episode_reward_mean"] >= args.stop_reward:
            os.makedirs(args.checkpoint_dir, exist_ok=True)
            trainer.save_checkpoint(args.checkpoint_dir)
            trainer.stop()
            break

    print(pretty_print(result))
    ray.shutdown()
