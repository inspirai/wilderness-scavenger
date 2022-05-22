import argparse
import sys

parser = argparse.ArgumentParser()
parser.add_argument("-T", "--timeout", type=int, default=60 * 5)  # The time length of one game (sec)
parser.add_argument("-R", "--time-scale", type=int, default=10)
parser.add_argument("-M", "--map-id", type=int, default=1)
parser.add_argument("-S", "--random-seed", type=int, default=0)
parser.add_argument("--target-location", type=float, nargs=3, default=[0, 0, 0])
parser.add_argument("--start-location", type=float, nargs=3, default=[0, 0, 0])
parser.add_argument("--base-worker-port", type=int, default=50000)
parser.add_argument("--start-hight", type=float, default=5)
parser.add_argument("--engine-dir", type=str, default="../wildscav-linux-backend")
parser.add_argument("--map-dir", type=str, default="../map_data")
parser.add_argument("--num-workers", type=int, default=10)
parser.add_argument("--eval-interval", type=int, default=10)
parser.add_argument("--record", action="store_true")
parser.add_argument("--replay-suffix", type=str, default="")
parser.add_argument("--checkpoint-dir", type=str, default="checkpoints_track1")
parser.add_argument("--detailed-log", action="store_true", help="whether to print detailed logs")
parser.add_argument("--run", type=str, default="ppo", help="The RLlib-registered algorithm to use.")
parser.add_argument("--stop-iters", type=int, default=300)
parser.add_argument("--stop-timesteps", type=int, default=1e8)
parser.add_argument("--stop-reward", type=float, default=95)
parser.add_argument("--use-depth", action="store_true")
parser.add_argument("--stop-episodes", type=float, default=50000)

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
    eval_cfg = vars(args).copy()
    eval_cfg["in_evaluation"]=True
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
            "evaluation_num_workers": 10,
            "evaluation_config":{"env_config": eval_cfg},
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
            "num_gpus":0,
            "evaluation_num_workers": 10,
            "evaluation_config":{"env_config": eval_cfg},
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
            "num_gpus":0,
            "evaluation_num_workers": 10,
            "evaluation_config":{"env_config": eval_cfg},
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
            "num_gpus":0,
            "evaluation_num_workers": 10,
            "evaluation_config":{"env_config": eval_cfg},
        }
    )
    else:
        raise ValueError('No such algorithm')
    step=0
    while True:
        step+=1
        result = trainer.train()
        print(f"current_training_steps:{step},current_alg:{alg}")
        if step !=0 and step %500==0:
            os.makedirs(args.checkpoint_dir+f"{alg}"+str(args.map_id), exist_ok=True)
            trainer.save_checkpoint(args.checkpoint_dir+f"{alg}"+str(args.map_id))
            print("trainer save a checkpoint")
        if result["episodes_total"] >= args.stop_episodes:
            os.makedirs(args.checkpoint_dir+f"{alg}"+str(args.map_id), exist_ok=True)
            trainer.save_checkpoint(args.checkpoint_dir+f"{alg}"+str(args.map_id))
            trainer.stop()
            break

    print("the training has done!!")
    ray.shutdown()
    sys.exit()


