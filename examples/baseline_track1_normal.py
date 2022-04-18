import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-T", "--timeout", type=int, required=True)
parser.add_argument("-R", "--time-scale", type=int, default=100)
parser.add_argument("-M", "--map-id", type=int, default=1)
parser.add_argument("-S", "--random-seed", type=int, default=0)
parser.add_argument("--trigger-range", type=float, default=1)
parser.add_argument("--target-location", type=float, nargs=3, default=[1, 0, 1])
parser.add_argument("--start-location", type=float, nargs=3, default=[0, 0, 0])
parser.add_argument("--start-range", type=float, default=2)
parser.add_argument("--start-hight", type=float, default=5)
parser.add_argument("--unity-path", type=str, default="../unity3d/fps.x86_64")
parser.add_argument("--unity-log-dir", type=str, default=".")
parser.add_argument("--use-depth", action="store_true")
parser.add_argument("--num-workers", type=int, default=0)
parser.add_argument("--eval-interval", type=int, default=None)
parser.add_argument("--run", type=str, default="PPO", help="The RLlib-registered algorithm to use.")
parser.add_argument("--stop-iters", type=int, default=9999)
parser.add_argument("--stop-timesteps", type=int, default=100000000)
parser.add_argument("--stop-reward", type=float, default=95)


if __name__ == "__main__":
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
            trainer.stop()
            break

    print(pretty_print(result))
    ray.shutdown()
