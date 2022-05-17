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
parser.add_argument("--engine-dir", type=str, default="../wildscav-linux-backend")
parser.add_argument("--map-dir", type=str, default="../map_data")
parser.add_argument("--num-workers", type=int, default=0)
parser.add_argument("--eval-interval", type=int, default=None)
parser.add_argument("--record", action="store_true")
parser.add_argument("--replay-suffix", type=str, default="")
parser.add_argument("--checkpoint-dir", type=str, default="checkpoints_track1_ppo")
parser.add_argument("--detailed-log", action="store_true", help="whether to print detailed logs")
parser.add_argument("--run", type=str, default="PPO", help="The RLlib-registered algorithm to use.")
parser.add_argument("--stop-iters", type=int, default=1000)
parser.add_argument("--stop-timesteps", type=int, default=100000000)
parser.add_argument("--stop-reward", type=float, default=95)


if __name__ == "__main__":
    import os
    import ray
    from ray.tune.logger import pretty_print
    from ray.rllib.agents.ppo import PPOTrainer
    from ray.rllib.agents.a3c import A3CTrainer
    # from ray.rllib.agents.sac import SACTrainer
    from ray.rllib.agents.impala import ImpalaTrainer
    # from ray.rllib.agents.ddpg import DDPGTrainer
    # from ray.rllib.agents.dqn.apex import ApexTrainer
    from ray.rllib.agents.ppo.appo import APPOTrainer
    from ray.rllib.agents.ppo.ddppo import DDPPOTrainer
    from envs.envs_track1 import NavigationBaseEnv

    args = parser.parse_args()

    ray.init()

    alg = args.run
    if alg =='ppo':
        trainer = PPOTrainer(
        config={
            "env": NavigationBaseEnv,
            "env_config": vars(args),
            "framework": "torch",
            "num_workers": args.num_workers,
            "evaluation_interval": args.eval_interval,
        }
    )
    elif alg=='appo':
        trainer = APPOTrainer(
        config={
            "env": NavigationBaseEnv,
            "env_config": vars(args),
            "framework": "torch",
            "num_workers": args.num_workers,
            "evaluation_interval": args.eval_interval,
            "num_gpus":0
        }
    )
    elif alg=='a3c':
        trainer = A3CTrainer(
        config={
            "env": NavigationBaseEnv,
            "env_config": vars(args),
            "framework": "torch",
            "num_workers": args.num_workers,
            "evaluation_interval": args.eval_interval,
            "num_gpus":0
        }
    )
    elif alg=='impala':
        trainer = ImpalaTrainer(
        config={
            "env": NavigationBaseEnv,
            "env_config": vars(args),
            "framework": "torch",
            "num_workers": args.num_workers,
            "evaluation_interval": args.eval_interval,
            "num_gpus":0
        }
    )
    else:
        raise ValueError('No such algorithm')
    while True:
        result = trainer.train()
        print(pretty_print(result))

        if result["episode_reward_mean"] >= args.stop_reward:
            os.makedirs(args.checkpoint_dir, exist_ok=True)
            trainer.save_checkpoint(args.checkpoint_dir)
            trainer.stop()
            break

    print("the training has done!!")
    ray.shutdown()


