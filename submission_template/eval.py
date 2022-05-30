REMOTE_MAP_DIR = "/root/map-data"
REMOTE_ENGINE_DIR = "/root/fps_linux_eval"
LOCAL_MAP_DIR = "/mnt/d/Codes/cog-local/map-data"
LOCAL_ENGINE_DIR = "/mnt/d/Codes/cog-local/fps_linux_train"


if __name__ == "__main__":
    import os
    import argparse
    from rich.pretty import pprint

    parser = argparse.ArgumentParser(description="Run evaluation for different tracks")
    parser.add_argument("--track", type=str, default="1a", help="Track to evaluate")
    parser.add_argument("--map-list", type=int, nargs="+", default=[1])
    parser.add_argument("--map-dir", type=str, default=LOCAL_MAP_DIR)
    parser.add_argument("--engine-dir", type=str, default=LOCAL_ENGINE_DIR)
    parser.add_argument("--episodes-per-map", type=int, default=10)
    parser.add_argument("--episode-timeout", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--local-test", action="store_true")
    args = parser.parse_args()

    eval_id = int(os.environ.get("EVAL_ID", 0))

    print(f">>>>>>>>>>>>>>>>>> Evaluation ID: {eval_id} <<<<<<<<<<<<<<<<<<<<")

    if not args.local_test:
        args.map_dir = REMOTE_MAP_DIR
        args.engine_dir = REMOTE_ENGINE_DIR
        print(f">>>>>> Copying evaluation scripts")
        os.system("cp /root/submission_template/*.py . && ls")
        print(f"<<<<<< Finished")

    if args.track == "1a":
        from eval_track_1_1 import run_eval
        if not args.local_test:
            args.map_list = list(range(101, 111))
    elif args.track == "1b":
        from eval_track_1_2 import run_eval
        if not args.local_test:
            args.map_list = list(range(111, 121))
    elif args.track == "2":
        from eval_track_2 import run_eval
        if not args.local_test:
            args.map_list = list(range(121, 131))
    else:
        raise ValueError(f"Unknown track {args.track}")

    pprint(args)

    from common import send_results, RunningStatus

    try:
        run_eval(args, eval_id)
        message = send_results({"id": eval_id, "status": RunningStatus.STOPPED})
    except Exception as e:
        message = send_results({"id": eval_id, "status": RunningStatus.ERROR})

    print("Response:", message)
