# game configs -- depth map w/h ratio is 16:9
DEPTH_MAP_FAR = 200
DEPTH_MAP_WIDTH = 64
DEPTH_MAP_HEIGHT = 36
TURN_ON_RECORDING = False


class RunningStatus:
    PENDING = 0
    STARTED = 1
    FINISHED = 2
    STOPPED = 3
    ERROR = 5


DEFAULT_PAYLOAD = {
    "id": None,
    "status": RunningStatus.PENDING,
    "current_episode": 0,
    "total_episodes": 0,
    "average_time_use": 0,
    "average_time_punish": 0,
    "average_time_total": 0,
    "success_rate": 0,
    "average_supply": 0,
}


def get_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--map-list", type=int, nargs="+", default=list(range(91, 100)))
    parser.add_argument("--map-dir", type=str, default="/data/map-data")
    parser.add_argument("--engine-dir", type=str, default="/data/fps_linux")
    parser.add_argument("--episodes-per-map", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-id", type=int, required=True)
    return parser.parse_args()


def send_results(data):
    import requests
    import logging
    
    url_head = "https://wildscav-eval.inspirai.com/api/evaluations/status?token=baiyangshidai_inspir"
    url = url_head + "&" + "&".join([f"{k}={v}" for k, v in data.items()])
    
    try:
        with requests.Session() as s:
            message = s.get(url, timeout=5).text
    except Exception as e:
        message = "Failed to send results"

    logger = logging.getLogger("rich")
    logger.info({
        "URL": url,
        "Response": message,
    })


if __name__ == "__main__":
    send_results(
        {
            "id": 7,
            "status": RunningStatus.STARTED,
            "current_episode": 5,
            "total_episodes": 10,
            "average_time_use": 0.1,
            "average_time_punish": 0.2,
            "average_time_total": 0.3,
            "success_rate": 0.4,
            "num_supply": 0,
        }
    )


    import logging
    from rich.logging import RichHandler

    FORMAT = "%(message)s"
    logging.basicConfig(
        level="NOTSET", format=FORMAT, datefmt="[%X]", handlers=[RichHandler()]
    )

    logger = logging.getLogger("rich")
    logger.setLevel("ERROR")
    logger.info("Hello, World!")
