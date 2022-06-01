import argparse
import re
import sys

parser = argparse.ArgumentParser()
parser.add_argument("-T", "--timeout", type=int, default=60 * 3)  # The time length of one game (sec)
parser.add_argument("-R", "--time-scale", type=int, default=10)
parser.add_argument("-M", "--map-id", type=int, default=1)
parser.add_argument("-S", "--random-seed", type=int, default=0)
parser.add_argument("--start-location", type=float, nargs=3, default=[0, 0, 0])
parser.add_argument("--target-location", type=float, nargs=3, default=[0, 0, 0])
parser.add_argument("--base-worker-port", type=int, default=50000)
parser.add_argument("--engine-dir", type=str, default="../wildscav-linux-backend")
parser.add_argument("--map-dir", type=str, default="../101_104")
parser.add_argument("--num-workers", type=int, default=10)
parser.add_argument("--eval-interval", type=int, default=None)
parser.add_argument("--record", action="store_true")
parser.add_argument("--replay-suffix", type=str, default="")
parser.add_argument("--checkpoint-dir", type=str, default="checkpoints-")
parser.add_argument("--detailed-log", action="store_true", help="whether to print detailed logs")
parser.add_argument("--run", type=str, default="ppo", help="The RLlib-registered algorithm to use.")
parser.add_argument("--stop-iters", type=int, default=300)
parser.add_argument("--stop-timesteps", type=int, default=1e8)
parser.add_argument("--stop-reward", type=float, default=98)
parser.add_argument("--use-depth", action="store_true")
parser.add_argument("--stop-episodes", type=float, default=200000)
parser.add_argument("--dmp-width", type=int, default=42)
parser.add_argument("--dmp-height", type=int, default=42)
parser.add_argument("--dmp-far", type=int, default=200)
parser.add_argument("--train-batch-size", type=int, default=2000)
parser.add_argument("--reload", type=bool, default=False)
parser.add_argument("--reload-dir", type=str, default="")
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
    from env_train import NavigationEnv

    args = parser.parse_args()
    eval_cfg = vars(args).copy()
    eval_cfg["in_evaluation"] = True
    ray.init()
    alg = args.run
    if alg == 'ppo':
        trainer = PPOTrainer(
            config={
                "env": NavigationEnv,
                "env_config": vars(args),
                "framework": "torch",
                "num_workers": args.num_workers,
                "train_batch_size": args.train_batch_size,  # default of ray is 4000
                "num_gpus": 0,
                "ignore_worker_failures":True,
            }
        )
    elif alg == 'a3c':
        trainer = A3CTrainer(
            config={
                "env": NavigationEnv,
                "env_config": vars(args),
                "framework": "torch",
                "num_workers": args.num_workers,
                "num_gpus": 0,
                "train_batch_size": args.train_batch_size,  # default of ray is 4000
                "ignore_worker_failures":True,

            }
        )
    elif alg == 'impala':
        trainer = ImpalaTrainer(
            config={
                "env": NavigationEnv,
                "env_config": vars(args),
                "framework": "torch",
                "num_workers": args.num_workers,
                "num_gpus": 0,
                "train_batch_size": args.train_batch_size,  # default of ray is 4000
                "ignore_worker_failures":True,
            }
        )
    else:
        raise ValueError('No such algorithm')
    step = 0
    if args.reload:
        trainer.restore(args.reload_dir)

    while True:
        step += 1
        result = trainer.train()
        reward = result["episode_reward_mean"]
        e = result["episodes_total"]
        len1 = result["episode_len_mean"]
        s = result["agent_timesteps_total"]
        print(f"current_alg:{alg},current_training_steps:{s},episodes_total:{e},current_reward:{reward},current_len:{len1}")

        if step != 0 and step % 200 == 0:
            os.makedirs(args.checkpoint_dir + f"{alg}" + str(args.map_id), exist_ok=True)
            trainer.save(args.checkpoint_dir + f"{alg}" + str(args.map_id))
            print("trainer save a checkpoint")
        if result["current_alg:{alg},"] >= args.stop_timesteps:
            os.makedirs(args.checkpoint_dir + f"{alg}" + str(args.map_id), exist_ok=True)
            trainer.save(args.checkpoint_dir + f"{alg}" + str(args.map_id))
            trainer.stop()
            break

    print("the training has done!!")
    ray.shutdown()
    sys.exit()


