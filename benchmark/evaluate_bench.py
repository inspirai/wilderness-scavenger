import argparse
import re
import sys

parser = argparse.ArgumentParser()
parser.add_argument("-T", "--timeout", type=int, default=60 * 2)  # The time length of one game (sec)
parser.add_argument("-R", "--time-scale", type=int, default=10)
parser.add_argument("-M", "--map-id", type=int, default=102)
parser.add_argument("-S", "--random-seed", type=int, default=0)
parser.add_argument("--start-location", type=float, nargs=3, default=[0, 0, 0])
parser.add_argument("--target-location", type=float, nargs=3, default=[0, 0, 0])
parser.add_argument("--base-worker-port", type=int, default=50000)
parser.add_argument("--engine-dir", type=str, default="../wildscav-linux-backend-v1.0-benchmark")
parser.add_argument("--map-dir", type=str, default="../101_104")
parser.add_argument("--num-workers", type=int, default=1)
parser.add_argument("--eval-interval", type=int, default=1)
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

    from ray.rllib.agents.sac import SACTrainer
    from env_train import NavigationEnv
    import numpy as np
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
                # "create_env_on_driver":True,
                # "evaluation_interval":1,

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
    print("done")
    trainer.restore(args.checkpoint_dir)
    env = NavigationEnv(eval_cfg)
    episode_len = []
    for i in range(20):
        state = env.reset()
        done = False
        step=0
        while not done:
            step+=1
            action = trainer.compute_single_action(state,explore=False)
            next_state,reward,done,info = env.step(action)
            state = next_state
        print(step)
        episode_len.append(step)
    acc = 1. - episode_len.count(1201)/len(episode_len)
    mean_ = np.mean(episode_len)
    std_ = np.std(episode_len)
    print(acc,mean_,std_)

    env.close()
    ray.shutdown()
    sys.exit()
    # trainer.compute_action
    # result = trainer.evaluate()
    # print(pretty_print(result))




