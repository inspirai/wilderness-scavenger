import json
import numpy as np
from typing import Any, Dict, Iterable
from rich.console import Console
from rich.table import Table


def load_json(file_path):
    with open(file_path, "r") as f:
        data = json.loads(f.read())
    return data


def get_picth_yaw(x, y, z):
    pitch = np.arctan2(y, (x**2 + z**2)**0.5) / np.pi * 180
    yaw = np.arctan2(x, z) / np.pi * 180
    return pitch, yaw


def get_distance(start, target):
    p0 = np.asarray(start)
    p1 = np.asarray(target)
    return np.linalg.norm(p1 - p0)


def get_position(state):
    return [
        state.position_x,
        state.position_y,
        state.position_z,
    ]


def vector3d_to_list(vec3d):
    return [vec3d.x, vec3d.y, vec3d.z]


def get_orientation(state):
    return [
        0,
        state.pitch,
        state.yaw,
    ]


def set_vector3d(vec3d, arr):
    vec3d.x = arr[0]
    vec3d.y = arr[1]
    vec3d.z = arr[2]


def set_GM_command(gm_cmd, config: Dict[str, Any]):
    for key, value in config.items():
        field = getattr(gm_cmd, key)
        if isinstance(field, (int, float, str)):
            setattr(gm_cmd, key, value)
        elif isinstance(field, Iterable):
            for obj in value:
                element = field.add()
                set_GM_command(element, obj)
        else:
            set_GM_command(field, value)


class ResultLogger:

    def __init__(self):
        self.console = Console()
        self.monitor_metrics = [
            (["training_iteration"], 0),
            (["timesteps_total"], 0),
            (["episode_reward_min"], 4),
            (["episode_reward_max"], 4),
            (["episode_reward_mean"], 4),
            (["info", "learner", "default_policy", "learner_stats",
              "entropy"], 4),
            (["info", "learner", "default_policy", "learner_stats", "kl"], 4),
        ]

    def get_metric_value(self, result, keys):
        if len(keys) == 1:
            return result[keys[0]]
        return self.get_metric_value(result[keys[0]], keys[1:])

    def print_result(self, res):
        table = Table(show_header=True, header_style="bold magenta")
        for metric, _ in self.monitor_metrics:
            table.add_column(metric[-1])
        val_str_list = [
            f"{round(self.get_metric_value(res, metric), prec)}"
            for metric, prec in self.monitor_metrics
        ]
        table.add_row(*val_str_list)
        self.console.print(table)
