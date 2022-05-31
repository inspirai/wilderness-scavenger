import random
import numpy as np
from typing import NamedTuple
from inspirai_fps.gamecore import AgentState
from inspirai_fps.utils import get_position, get_picth_yaw


# DO NOT MODIFY THIS CLASS
class NavigationAction(NamedTuple):
    walk_dir: float
    walk_speed: float
    turn_lr_delta: float
    look_ud_delta: float
    jump: bool


# DO NOT MODIFY THIS CLASS
class SupplyGatherAction(NamedTuple):
    walk_dir: float
    walk_speed: float
    turn_lr_delta: float
    look_ud_delta: float
    jump: bool
    pickup: bool


# DO NOT MODIFY THIS CLASS
class SupplyBattleAction(NamedTuple):
    walk_dir: float
    walk_speed: float
    turn_lr_delta: float
    look_ud_delta: float
    jump: bool
    pickup: bool
    attack: bool
    reload: bool


class AgentNavigation:
    """
    This is a template of an agent for the navigation task.
    TODO: Modify the code in this class to implement your agent here.
    """

    def __init__(self, episode_info) -> None:
        self.episode_info = episode_info

    def act(self, ts: int, state: AgentState) -> NavigationAction:
        pos = np.asarray(get_position(state))
        tar = np.asarray(self.episode_info["target_location"])
        dir = tar - pos
        dir = dir / np.linalg.norm(dir)
        walk_dir = get_picth_yaw(*dir)[1] % 360

        return NavigationAction(
            walk_dir=walk_dir,
            walk_speed=5,
            turn_lr_delta=0,
            look_ud_delta=0,
            jump=False,
        )


class AgentSupplyGathering:
    """
    This is a template of an agent for the supply gathering task.
    TODO: Modify the code in this class to implement your agent here.
    """

    def __init__(self, episode_info) -> None:
        self.episode_info = episode_info

    def act(self, ts: int, state: AgentState) -> SupplyGatherAction:
        pos = np.asarray(get_position(state))
        if state.supply_states:
            supply_info = list(state.supply_states.values())[0]
            tar = np.asarray(get_position(supply_info))
            dir = tar - pos
            dir = dir / np.linalg.norm(dir)
            walk_dir = get_picth_yaw(*dir)[1] % 360
        else:
            walk_dir = random.randint(0, 360)

        return SupplyGatherAction(
            walk_dir=walk_dir,
            walk_speed=5,
            turn_lr_delta=0,
            look_ud_delta=0,
            jump=False,
            pickup=True,
        )


class AgentSupplyBattle:
    """
    This is a template of an agent for the supply battle task.
    TODO: Modify the code in this class to implement your agent here.
    """

    def __init__(self, episode_info) -> None:
        self.episode_info = episode_info

    def act(self, ts: int, state: AgentState) -> SupplyBattleAction:
        pos = np.asarray(get_position(state))
        if state.supply_states:
            supply_info = list(state.supply_states.values())[0]
            tar = np.asarray(get_position(supply_info))
            dir = tar - pos
            dir = dir / np.linalg.norm(dir)
            walk_dir = get_picth_yaw(*dir)[1] % 360
        else:
            walk_dir = random.randint(0, 360)

        turn_lr_delta = 0
        look_ud_delta = 0
        attack = False

        if state.enemy_states:
            enemy_info = list(state.enemy_states.values())[0]
            tar = np.asarray(get_position(enemy_info))
            dir = tar - pos
            dir = dir / np.linalg.norm(dir)
            aim_pitch, aim_yaw = get_picth_yaw(*dir)

            diff_pitch = aim_pitch - state.pitch
            diff_yaw = aim_yaw - state.yaw
            if abs(diff_pitch) < 5 and abs(diff_yaw) < 5:
                attack = True

            skip_frames = self.episode_info["time_step_per_action"]
            rotate_speed_decay = 0.5
            turn_lr_delta = diff_yaw / skip_frames * rotate_speed_decay
            look_ud_delta = diff_pitch / skip_frames * rotate_speed_decay

        return SupplyBattleAction(
            walk_dir=walk_dir,
            walk_speed=5,
            turn_lr_delta=turn_lr_delta,
            look_ud_delta=look_ud_delta,
            jump=False,
            pickup=True,
            attack=attack,
            reload=state.weapon_ammo < 5 and state.spare_ammo > 0,
        )
