import os
import sys
import grpc
import time
import random
import datetime
import numpy as np
import subprocess
from queue import Queue
from concurrent import futures
from typing import Dict, List, Tuple, Any

from google.protobuf.json_format import MessageToDict

from inspirai_fps import simple_command_pb2
from inspirai_fps import simple_command_pb2_grpc
from inspirai_fps.raycast_manager import RaycastManager
from inspirai_fps.utils import (
    get_orientation,
    get_position,
    set_vector3d,
    set_GM_command,
    load_json,
    vector3d_to_list,
    get_distance,
)


__all__ = [
    "ActionVariable",
    "Game",
]


class QueueServer(simple_command_pb2_grpc.CommanderServicer):
    def __init__(self, request_queue, reply_queue) -> None:
        super().__init__()
        self.request_queue = request_queue
        self.reply_queue = reply_queue

    def Request_S2A_UpdateGame(self, request, context):
        # print(request)
        self.request_queue.put(request)
        reply = self.reply_queue.get()
        # print(reply)
        return reply


class StateVariable:
    LOCATION = "location"
    MOVE_DIR = "move_dir"
    MOVE_SPEED = "move_speed"
    CAMERA_DIR = "camera_dir"
    HEALTH = "hp"
    WEAPON_AMMO = "num_gun_ammo"
    SPARE_AMMO = "num_pack_ammo"
    ON_GROUND = "on_ground"
    IS_ATTACKING = "is_attack"
    IS_RELOADING = "is_reload"
    HIT_ENEMY = "hit_enemy"
    HIT_ENEMY_ID = "hit_enemy_id"
    HIT_BY_ENEMY = "hit_by_enemy"
    HIT_BY_ENEMY_ID = "hit_by_enemy_id"
    NUM_SUPPLY = "num_supply"
    IS_WAITING_RESPAWN = "is_waiting_respawn"
    IS_INVISIBLE = "is_invisible"


class ActionVariable:
    WALK_DIR = "walk_dir"
    WALK_SPEED = "walk_speed"
    TURN_LR_DELTA = "turn_left_right_delta"
    LOOK_UD_DELTA = "look_up_down_delta"
    JUMP = "jump"
    ATTACK = "shoot_gun"
    RELOAD = "reload"
    PICKUP = "collect"


class SupplyState:
    """show detailed supply information, including:
    - position_[xyz]: 3d space coordinates of the supply refresh point
    - quantity: the number of supplies at that point
    """

    def __init__(self, supply_info) -> None:
        self.position_x = supply_info.supply_location.x
        self.position_y = supply_info.supply_location.y
        self.position_z = supply_info.supply_location.z
        self.quantity = supply_info.supply_quantity
        self.id = supply_info.supply_id

    def __repr__(self) -> str:
        x = self.position_x
        y = self.position_y
        z = self.position_z
        id = self.id
        q = self.quantity
        return f"Supply({id=})_Pos(x={x:.2f},y={y:.2f},z={z:.2f})_Num({q})"


class EnemyStateDetailed:
    """show detailed information of the enemy, including:
    - position_[xyz]: 3d space coordinates of the enemy
    - move_dir_[xyz]: moving direction of the enemy (which is a 3d unit vector)
    - move_speed: moving speed of the agent in the 3d space
    """

    def __init__(self, enemy_info: simple_command_pb2.EnemyInfo) -> None:
        self.position_x = enemy_info.location.x
        self.position_y = enemy_info.location.y
        self.position_z = enemy_info.location.z
        # self.move_dir_x = enemy_info.move_dir.x
        # self.move_dir_y = enemy_info.move_dir.y
        # self.move_dir_z = enemy_info.move_dir.z
        # self.move_speed = enemy_info.move_speed
        self.health = enemy_info.hp
        self.id = enemy_info.enemy_id
        self.waiting_respawn = enemy_info.is_respawn
        self.is_invincible = enemy_info.is_invincible

    def __repr__(self) -> str:
        x = self.position_x
        y = self.position_y
        z = self.position_z
        # mx = self.move_dir_x
        # my = self.move_dir_y
        # mz = self.move_dir_z
        # ms = self.move_speed
        h = self.health
        id = self.id
        w = self.waiting_respawn
        return f"Enemy({id=})_Pos({x=:.2f},{y=:.2f},{z=:.2f})_Health({h:.2f})_WaitingRespawn({w})_Invincible({self.is_invincible})"


class EnemyStateRough:
    """show simplified information of the enemy, including:
    - dir_vec: a 2d unit vector pointing from the agent's location to the enemy's location
    """

    def __init__(
        self,
        enemy_info: simple_command_pb2.EnemyInfo,
        obs_data: simple_command_pb2.Observation,
    ) -> None:
        self_pos_x = obs_data.location.x
        self_pos_z = obs_data.location.z
        enemy_pos_x = enemy_info.location.x
        enemy_pos_z = enemy_info.location.z

        dx = enemy_pos_x - self_pos_x
        dz = enemy_pos_z - self_pos_z
        dir_vec = np.asarray([dx, dz])

        self.dir_vec = dir_vec / np.linalg.norm(self.dir_vec)

    def __reduce__(self) -> str or Tuple[Any, ...]:
        return f"EnemyStateDirOnly_EnemyDir({self.dir_vec.tolist()})"


class AgentState:
    """
    wrap all perceivable information from the agent, including:
    - position_[xyz] `List[float]`: the 3d space coordinates (unit: M)
    - move_dir_[xyz] `List[float]`: the 3d space moving direction (which is a 3-dim unit vector)
    - move_speed `float`: the moving speed of the agent in the 3d space (unit: M/S)
    - pitch `float`: the horizontal angle of the agent's camera relative to the Northward direction
    - yaw `float`: the vertical angle of the agent's camera relative to the horizontal line
    - health `int`: the HP of the agent (if decreased to 0 then the agent dies)
    - weapon_ammo `int`: the number of bullets left in the agent's weapon
    - spare_ammo `int`: the number of bullets left for refilling weapon ammo
    - on_ground `bool`: whether the agent is on the ground or dangling
    - is_attack `bool`: whether the agent is shooting with the weapon
    - is_reload `bool`: whether the agent is refilling the weapon ammo
    - hit_enemy `bool`: whether the agent's weapon fire hits an enemy (e.g. another agent)
    - hit_by_enemy `bool`: whether the agent is hit by a weapon fire
    - num_supply `int`: the total amount of supplies that the agent has collected
    - is_waiting_respawn `bool`: whether the agent is dead and waiting for a respawn
    - is_invincible `bool`: whether the agent is in INVINCIBLE state which means no harm will take effects
    - depth_map `Iterable[Iterable[float]]`: an H x W array representing the depth of mesh point in the agent's visual-sight window (if turned off this is `None`)
    """

    __SUPPLY_VIS_DISTANCE = 20
    __ENEMY_VISIBLE_ANGLE = 90
    __CAMERA_HEIGHT = 1.5
    __BODY_RADIUS = 0.45
    __BODY_HEIGHT = 1.78

    def __init__(
        self,
        obs_data,
        ray_tracer,
        use_depth_map=False,
    ) -> None:
        self.position_x = obs_data.location.x
        self.position_y = obs_data.location.y
        self.position_z = obs_data.location.z
        self.move_dir_x = obs_data.move_dir.x
        self.move_dir_y = obs_data.move_dir.y
        self.move_dir_z = obs_data.move_dir.z
        self.move_speed = obs_data.move_speed
        self.pitch = obs_data.pitch  # [-90, 90]
        self.yaw = (
            obs_data.yaw if obs_data.yaw <= 180 else obs_data.yaw - 360
        )  # (-180, 180]
        self.health = obs_data.hp
        self.weapon_ammo = obs_data.num_gun_ammo
        self.spare_ammo = obs_data.num_pack_ammo
        self.on_ground = obs_data.on_ground
        self.is_attack = obs_data.is_fire
        self.is_reload = obs_data.is_reload
        self.hit_enemy = obs_data.hit_enemy
        self.hit_enemy_id = obs_data.hit_enemy_id
        self.hit_by_enemy = obs_data.hit_by_enemy
        self.hit_by_enemy_id = obs_data.hit_by_enemy_id
        self.num_supply = obs_data.num_supply
        self.is_waiting_respawn = obs_data.is_waiting_respawn
        self.is_invincible = obs_data.is_invincible

        self.ray_tracer = ray_tracer

        self.depth_map = None
        if use_depth_map:
            pos = [
                self.position_x,
                self.position_y + self.__CAMERA_HEIGHT,
                self.position_z,
            ]
            dir = get_orientation(self)
            self.depth_map = self.ray_tracer.get_depth(pos, dir)[0]

        self.supply_states = {
            s.supply_id: SupplyState(s)
            for s in filter(self.is_supply_visible, obs_data.supply_info_list)
        }

        self.enemy_states = {
            e.enemy_id: EnemyStateDetailed(e)
            for e in filter(self.is_enemy_visible, obs_data.enemy_info_list)
        }

    def __repr__(self) -> str:
        x = self.position_x
        y = self.position_y
        z = self.position_z
        pos = [round(p, 2) for p in (x, y, z)]

        dx = self.move_dir_x
        dy = self.move_dir_y
        dz = self.move_dir_z
        dir = [round(d, 2) for d in (dx, dy, dz)]

        return f"AgentState(pos={pos},dir={dir},speed={self.move_speed},pitch={self.pitch},yaw={self.yaw},health={self.health},weapon_ammo={self.weapon_ammo},spare_ammo={self.spare_ammo},on_ground={self.on_ground},is_attack={self.is_attack},is_reload={self.is_reload},hit_enemy_id={self.hit_enemy_id},hit_by_enemy_id={self.hit_by_enemy_id},num_supply={self.num_supply},is_waiting_respawn={self.is_waiting_respawn},is_invincible={self.is_invincible},use_depth_map={self.depth_map is not None})"

    def is_enemy_visible(self, enemy_info: simple_command_pb2.EnemyInfo):
        view_angle = [self.__ENEMY_VISIBLE_ANGLE / 16 * 9, self.__ENEMY_VISIBLE_ANGLE]
        body_param = [
            self.__BODY_RADIUS * 2,
            self.__BODY_HEIGHT,
        ]  # body size (width, height)

        agent_team_id = [0, 1]
        agent_position = [
            self.position_x,
            self.position_y,
            self.position_z,
            enemy_info.location.x,
            enemy_info.location.y,
            enemy_info.location.z,
        ]
        camera_location = [
            self.position_x,
            self.position_y + self.__CAMERA_HEIGHT,
            self.position_z,
            0,  # default 0 -> of no use
            0,  # default 0 -> of no use
            0,  # default 0 -> of no use
        ]
        camerarotation = [
            0,
            self.pitch,
            self.yaw,
            0,  # default 0 -> of no use
            0,  # default 0 -> of no use
            0,  # default 0 -> of no use
        ]

        is_visible = self.ray_tracer.agent_is_visible(
            body_param,
            view_angle,
            agent_team_id,
            agent_position,
            camera_location,
            camerarotation,
        )

        return is_visible[0][1]

    def is_supply_visible(self, supply_info: simple_command_pb2.SupplyInfo):
        supply_pos = vector3d_to_list(supply_info.supply_location)
        self_pos = get_position(self)
        distance = get_distance(self_pos, supply_pos)
        return distance <= self.__SUPPLY_VIS_DISTANCE

    @property
    def supply_visible_distance(self):
        return self.__SUPPLY_VIS_DISTANCE

    @property
    def enemy_visible_angle(self):
        return self.__ENEMY_VISIBLE_ANGLE


class Game:
    # the game mode indicators
    MODE_NAVIGATION = simple_command_pb2.GameModeType.NAVIGATION_MODE
    MODE_SUP_GATHER = simple_command_pb2.GameModeType.SUP_GATHER_MODE
    MODE_SUP_BATTLE = simple_command_pb2.GameModeType.SUP_BATTLE_MODE

    # game config constants that are not changeable by the user
    __SPEED_UP_FACTOR = 10
    __TRIGGER_DISTANCE = 1
    __WATER_SPEED_DECAY = 0.5
    __INVINCIBLE_TIME = 10
    __RESPAWN_TIME = 10
    __SUPPLY_DROP_PERCENT = 50
    __TIMESTEP_PER_ACTION = 5
    __MAX_WALK_SPEED = 10
    __FRAME_RATE = 50

    def __init__(
        self,
        map_dir=None,
        engine_dir=None,
        engine_log_dir=None,
        server_port=50000,
        server_ip="127.0.0.1",
    ):
        self.__map_dir = map_dir
        self.__indoor_locations = None
        self.__outdoor_locations = None
        self.__available_actions = None
        self.__engine_dir = engine_dir
        self.__engine_log_dir = engine_log_dir
        self.__server_ip = server_ip
        self.__server_port = server_port
        self.__time_step = 0
        self.__latest_request = None
        self.__log_trajectory = False

        # initialize default game settings
        self.__GM = self.__get_default_GM()
        self.__use_depth_map = False

        # initialize default agent
        self.add_agent()

        # load default map
        self.set_map_id(1)

        # load default ray tracer
        mesh_name = f"{self.__GM.map_id:03d}.obj"
        mesh_file_path = os.path.join(self.__map_dir, mesh_name)
        self.__ray_tracer = RaycastManager(mesh_file_path)
        self.dmp_width = self.__ray_tracer.WIDTH
        self.dmp_height = self.__ray_tracer.HEIGHT
        self.dmp_far = self.__ray_tracer.FAR

    def __get_default_GM(self):
        gm_command = simple_command_pb2.GMCommand()

        # Common settings
        gm_command.timeout = 10
        gm_command.game_mode = Game.MODE_NAVIGATION
        gm_command.time_scale = Game.__SPEED_UP_FACTOR
        gm_command.random_seed = 0
        gm_command.num_agents = 0
        gm_command.is_record = False
        gm_command.replay_suffix = ""
        gm_command.water_speed_decay = Game.__WATER_SPEED_DECAY

        # ModeNavigation settings
        set_vector3d(gm_command.target_location, [1, 0, 1])
        gm_command.trigger_range = Game.__TRIGGER_DISTANCE

        # ModeSupplyGather settings
        set_vector3d(gm_command.supply_heatmap_center, [0, 0, 0])
        gm_command.supply_heatmap_radius = 1
        gm_command.supply_create_percent = 1
        gm_command.supply_house_create_percent = 1
        gm_command.supply_grid_length = 10
        gm_command.supply_random_min = 1
        gm_command.supply_random_max = 1
        gm_command.supply_house_random_min = 10
        gm_command.supply_house_random_max = 10

        # ModeSupplyBattle settings
        gm_command.respawn_time = Game.__RESPAWN_TIME
        gm_command.invincible_time = Game.__INVINCIBLE_TIME
        gm_command.supply_loss_percent_when_dead = Game.__SUPPLY_DROP_PERCENT

        return gm_command

    def get_game_result(self):
        """do not call this function before one episode ends"""
        game_state = self.__latest_request.game_state
        assert game_state == simple_command_pb2.GameState.over

        obs = self.__latest_request.agent_obs_list[0]

        if self.__GM.game_mode == Game.MODE_NAVIGATION:
            reach_target = False
            punish_time = 0

            loc = vector3d_to_list(obs.location)
            tar = vector3d_to_list(self.__GM.target_location)

            used_time = self.__time_step / self.__FRAME_RATE
            distance_to_target = get_distance(loc, tar)

            if (
                used_time < self.__GM.timeout
                or distance_to_target <= self.target_trigger_distance
            ):
                reach_target = True

            if not reach_target:
                punish_time = 2 * distance_to_target / self.__MAX_WALK_SPEED

            return {
                "reach_target": reach_target,
                "punish_time": punish_time,
                "used_time": used_time,
            }

        return {
            "num_supply": self.__num_supply,
        }

    @property
    def time_step_per_action(self):
        return self.__TIMESTEP_PER_ACTION

    @property
    def frame_rate(self):
        return self.__FRAME_RATE

    def set_game_config(self, config_path: str):
        """
        Experimental:
        ----------
        Set game config from a json file. Be careful with the config file format!
        """
        game_config = load_json(config_path)
        self.__GM.Clear()
        set_GM_command(self.__GM, game_config)

    def get_game_config(self):
        game_config = MessageToDict(self.__GM)
        game_config["use_depth_map"] = self.__use_depth_map
        if self.__use_depth_map:
            game_config["dmp_width"] = self.dmp_width
            game_config["dmp_height"] = self.dmp_height
            game_config["dmp_far"] = self.dmp_far
        return game_config

    def get_agent_name(self, agent_id):
        for agent in self.__GM.agent_setups:
            if agent.id == agent_id:
                return agent.agent_name

    def set_episode_timeout(self, timeout: int):
        assert isinstance(timeout, int) and timeout > 0
        self.__GM.timeout = timeout

    def set_game_mode(self, game_mode: int):
        assert game_mode in [0, 1, 2]
        self.__GM.game_mode = game_mode

    def set_map_id(self, map_id: int):
        self.__GM.map_id = map_id

        # load location data
        location_file_path = os.path.join(self.__map_dir, f"{map_id:03d}.json")
        locations = load_json(location_file_path)
        self.__indoor_locations = locations["indoor"]
        self.__outdoor_locations = locations["outdoor"]
        self.__valid_locations = locations
        print(f"Loaded valid locations from {location_file_path}")

    def get_valid_locations(self):
        return self.__valid_locations.copy()

    def set_random_seed(self, random_seed: int):
        assert isinstance(random_seed, int)
        self.__GM.random_seed = random_seed

    def set_start_location(self, loc: List[float], agent_id: int = 0):
        assert isinstance(agent_id, int) and 0 <= agent_id < len(self.__GM.agent_setups)
        assert isinstance(loc, list) and len(loc) == 3
        agent = self.__GM.agent_setups[agent_id]
        set_vector3d(agent.start_location, loc)

    def get_start_location(self, agent_id: int = 0):
        for agent in self.__GM.agent_setups:
            if agent.id == agent_id:
                return vector3d_to_list(agent.start_location)

    def random_start_location(self, agent_id: int = 0, indoor: bool = False):
        assert isinstance(agent_id, int) and 0 <= agent_id < len(self.__GM.agent_setups)
        agent = self.__GM.agent_setups[agent_id]
        locations = self.__indoor_locations if indoor else self.__outdoor_locations
        loc = random.choice(locations)
        set_vector3d(agent.start_location, loc)

    def set_target_location(self, loc: List[float]):
        assert isinstance(loc, list) and len(loc) == 3
        set_vector3d(self.__GM.target_location, loc)

    def get_target_location(self):
        return vector3d_to_list(self.__GM.target_location)

    def random_target_location(self, indoor: bool = False):
        locations = self.__indoor_locations if indoor else self.__outdoor_locations
        loc = random.choice(locations)
        set_vector3d(self.__GM.target_location, loc)

    def set_available_actions(self, actions: List[str]):
        assert isinstance(actions, list) and len(actions) >= 1
        self.__available_actions = actions
        self.__action_idx_map = {
            key: i for i, key in enumerate(self.__available_actions)
        }

    def get_available_actions(self):
        return self.__available_actions.copy()

    def set_game_replay_suffix(self, replay_suffix: str):
        assert isinstance(replay_suffix, str)
        self.__GM.replay_suffix = replay_suffix

    def set_supply_heatmap_center(self, loc: List[float]):
        """loc: a list of two numbers that represent the x and z values of the center location"""
        assert isinstance(loc, list) and len(loc) == 2
        center = [loc[0], 0, loc[1]]
        set_vector3d(self.__GM.supply_heatmap_center, center)

    def random_supply_heatmap_center(self, indoor: bool = True):
        locations = self.__indoor_locations if indoor else self.__outdoor_locations
        loc = random.choice(locations)
        set_vector3d(self.__GM.supply_heatmap_center, loc)

    def get_supply_heatmap_center(self):
        return vector3d_to_list(self.__GM.supply_heatmap_center)

    def set_supply_heatmap_radius(self, radius: int):
        assert isinstance(radius, int) and radius > 0
        self.__GM.supply_heatmap_radius = radius

    def get_supply_heatmap_radius(self):
        return self.__GM.supply_heatmap_radius

    def set_supply_outdoor_richness(self, richness: int):
        assert isinstance(richness, int) and 0 <= richness <= 50
        self.__GM.supply_create_percent = richness

    def set_supply_indoor_richness(self, richness: int):
        assert isinstance(richness, int) and 0 <= richness <= 100
        self.__GM.supply_house_create_percent = richness

    def set_supply_spacing(self, spacing: int):
        assert isinstance(spacing, int) and spacing >= 1
        self.__GM.supply_grid_length = spacing

    def set_supply_outdoor_quantity_range(self, qmin=1, qmax=1):
        assert isinstance(qmin, int) and isinstance(qmax, int) and 1 <= qmin <= qmax
        self.__GM.supply_random_min = qmin
        self.__GM.supply_random_max = qmax

    def set_supply_indoor_quantity_range(self, qmin=1, qmax=1):
        assert isinstance(qmin, int) and isinstance(qmax, int) and 1 <= qmin <= qmax
        self.__GM.supply_house_random_min = qmin
        self.__GM.supply_house_random_max = qmax

    def add_supply_refresh(
        self,
        refresh_time: int,
        heatmap_radius: int,
        heatmap_center: List[float],
        indoor_richness: int,
        outdoor_richness: int,
    ):
        assert isinstance(refresh_time, int) and refresh_time >= 1
        assert isinstance(heatmap_radius, int) and heatmap_radius > 0
        assert isinstance(heatmap_center, list) and len(heatmap_center) in [2, 3]
        assert isinstance(indoor_richness, int) and 0 <= indoor_richness <= 100
        assert isinstance(outdoor_richness, int) and 0 <= outdoor_richness <= 50

        refresh = self.__GM.supply_refresh_datas.add()
        if len(heatmap_center) == 2:
            center = [heatmap_center[0], 0, heatmap_center[1]]
        else:
            center = heatmap_center
        set_vector3d(refresh.supply_heatmap_center, center)
        refresh.supply_heatmap_radius = heatmap_radius
        refresh.supply_refresh_time = refresh_time
        refresh.supply_create_percent = outdoor_richness
        refresh.supply_house_create_percent = indoor_richness

    def clear_supply_refresh(self):
        del self.__GM.supply_refresh_datas[:]

    def add_agent(
        self,
        health=100,
        num_pack_ammo=60,
        num_clip_ammo=15,
        attack=20,
        start_location=[0, 0, 0],
        agent_name=None,
    ):
        assert isinstance(start_location, list) and len(start_location) == 3
        assert isinstance(health, int) and 1 <= health <= 100
        assert isinstance(num_clip_ammo, int) and num_clip_ammo >= 0
        assert isinstance(num_pack_ammo, int) and num_pack_ammo >= 0
        assert isinstance(attack, int) and attack >= 1

        agent = self.__GM.agent_setups.add()
        agent_name = agent_name or f"agent_{self.__GM.num_agents}"

        assert isinstance(agent_name, str) and len(agent_name) > 0

        agent.hp = health
        agent.num_pack_ammo = num_pack_ammo
        agent.gun_capacity = num_clip_ammo
        agent.attack_power = attack
        agent.agent_name = agent_name
        agent.id = self.__GM.num_agents
        set_vector3d(agent.start_location, start_location)

        self.__GM.num_agents += 1

    def turn_on_record(self):
        self.__GM.is_record = True

    def turn_off_record(self):
        self.__GM.is_record = False

    def turn_on_depth_map(self):
        self.__use_depth_map = True

    def turn_off_depth_map(self):
        self.__use_depth_map = False

    def get_depth_map_size(self):
        return self.__ray_tracer.WIDTH, self.__ray_tracer.HEIGHT, self.__ray_tracer.FAR

    def set_depth_map_size(self, width, height, far=None):
        assert isinstance(width, int) and width > 0
        assert isinstance(height, int) and height > 0
        assert isinstance(far, int) or far is None
        self.dmp_width = width
        self.dmp_height = height
        if far is not None:
            assert far > 0
            self.dmp_far = far

    def get_time_step(self):
        """Returns the current number of ticks in this episode of the game"""
        return self.__time_step

    @property
    def max_walk_speed(self):
        return self.__MAX_WALK_SPEED

    @property
    def target_trigger_distance(self):
        return self.__GM.trigger_range

    def init(self):
        self.request_queue = Queue()
        self.reply_queue = Queue()

        # initialize message queue server
        self.server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
        simple_command_pb2_grpc.add_CommanderServicer_to_server(
            QueueServer(self.request_queue, self.reply_queue), self.server
        )
        self.server.add_insecure_port(f"[::]:{self.__server_port}")
        self.server.start()
        print("Server started ...")

        if sys.platform.startswith("linux"):
            engine_path = os.path.join(self.__engine_dir, "fps.x86_64")
            os.system(f"chmod +x {engine_path}")
            cmd = f"{engine_path} -IP:{self.__server_ip} -PORT:{self.__server_port}"
        elif sys.platform.startswith("win32"):
            engine_path = os.path.join(self.__engine_dir, "FPSGameUnity.exe")
            cmd = (
                f"start {engine_path} -IP:{self.__server_ip} -PORT:{self.__server_port}"
            )
        elif sys.platform.startswith("darwin"):
            engine_path = self.__engine_dir
            assert engine_path.endswith(".app"), "engine_dir must be a .app on MacOS"
            cmd = f"open {engine_path} --args -IP:{self.__server_ip} -PORT:{self.__server_port}"
        else:
            raise NotImplementedError(f"Platform {sys.platform} is not supported")

        if self.__engine_log_dir:
            os.makedirs(self.__engine_log_dir, exist_ok=True)
            timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            engine_log_name = f"{timestamp}-port-{self.__server_port}.log"
            engine_log_path = os.path.join(self.__engine_log_dir, engine_log_name)
            f = open(engine_log_path, "w")
        else:
            f = subprocess.DEVNULL

        # start unity3d game engine
        shell = sys.platform.startswith("win32")
        self.engine_process = subprocess.Popen(
            cmd.split(), stdout=f, stderr=f, shell=shell
        )
        print("Unity3D started ...")

        # waiting for unity3d to send the first request
        self.request_queue.get()  # the first request is only used to activate the server
        print("Unity3D connected ...")

    def get_state(self, agent_id=0) -> AgentState:
        for obs_data in self.__latest_request.agent_obs_list:
            if obs_data.id == agent_id:
                return AgentState(
                    obs_data,
                    self.__ray_tracer,
                    self.__use_depth_map,
                )

    def get_state_all(self) -> Dict[int, AgentState]:
        state_dict = {}
        for obs_data in self.__latest_request.agent_obs_list:
            agent_id = obs_data.id
            state_dict[agent_id] = AgentState(
                obs_data, self.__ray_tracer, self.__use_depth_map
            )
        return state_dict

    def make_action(self, action_cmd_dict: Dict[int, List]):
        """
        Parameters
        ----------
        `action_cmd_dict`: dict of {`agent_id`: `action`}
        `action`: list of action variable values
        >>> {0: [30, 2, True, False]}  # (WALK_DIR, WALK_SPEED, JUMP, FIRE)
        """
        reply = simple_command_pb2.A2S_Reply_Data()
        reply.game_state = simple_command_pb2.GameState.update

        for agent_id, action in action_cmd_dict.items():
            agent_cmd = reply.agent_cmd_list.add()
            agent_cmd.id = agent_id
            for action_name, idx in self.__action_idx_map.items():
                a = action[idx]
                if action_name == ActionVariable.WALK_SPEED:
                    a = min(a, self.__MAX_WALK_SPEED)
                setattr(agent_cmd, action_name, a)

        self.reply_queue.put(reply)
        self.__update_request()
        self.__time_step += self.__TIMESTEP_PER_ACTION

    def make_action_by_list(self, action_all: Dict[int, List[Tuple[str, Any]]]):
        reply = simple_command_pb2.A2S_Reply_Data()
        reply.game_state = simple_command_pb2.GameState.update

        for agent_id, action_list in action_all.items():
            agent_cmd = reply.agent_cmd_list.add()
            agent_cmd.id = agent_id
            for action_name, value in action_list:
                setattr(agent_cmd, action_name, value)

        self.reply_queue.put(reply)
        self.__update_request()
        self.__time_step += self.__TIMESTEP_PER_ACTION

    def log_movement_trajectory(self):
        self.__log_trajectory = True
        self.__points = []

    def log_movement_trajectory_stop(self):
        self.__log_trajectory = False

    def get_movement_trajectory(self):
        return self.__points

    def __update_request(self):
        self.__latest_request = self.request_queue.get()
        if self.__latest_request.game_state == simple_command_pb2.GameState.over:
            self.__is_episode_finished = True
        else:
            self.__is_episode_finished = False
            self.__num_supply = self.__latest_request.agent_obs_list[0].num_supply

        if self.__log_trajectory:
            pos = vector3d_to_list(self.__latest_request.agent_obs_list[0].location)
            x, z = pos[0], pos[2]
            self.__points.append((x, z))

    def is_episode_finished(self):
        return self.__is_episode_finished

    def new_episode(self):
        reply = simple_command_pb2.A2S_Reply_Data()
        reply.game_state = simple_command_pb2.GameState.reset
        reply.gm_cmd.CopyFrom(self.__GM)
        self.reply_queue.put(reply)
        self.__update_request()
        self.__time_step = 0

        print("Started new episode ...")

        # load mesh data
        mesh_name = f"{self.__GM.map_id:03d}.obj"
        mesh_file_path = os.path.join(self.__map_dir, mesh_name)

        if self.__ray_tracer.mesh_file_path != mesh_file_path:
            # release raytracer resources
            del self.__ray_tracer
            self.__ray_tracer = RaycastManager(mesh_file_path)
            print("[change] Loaded map mesh from {}".format(mesh_file_path))
        else:
            print("[keep] Reused map mesh from {}".format(mesh_file_path))

        # setup depth map size
        self.__ray_tracer.WIDTH = self.dmp_width
        self.__ray_tracer.HEIGHT = self.dmp_height
        self.__ray_tracer.FAR = self.dmp_far

    def close(self):
        reply = simple_command_pb2.A2S_Reply_Data()
        reply.game_state = simple_command_pb2.GameState.close
        self.reply_queue.put(reply)
        time.sleep(1)
        self.server.stop(0)
        print("Server stopped ...")

        self.engine_process.kill()
        print("Unity3D killed ...")

    @property
    def use_depth_map(self):
        return self.__use_depth_map


if __name__ == "__main__":
    import argparse
    from rich.console import Console
    from rich.progress import track

    console = Console()

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=50051)
    parser.add_argument("--timeout", type=int, default=10)
    parser.add_argument("--game-mode", type=int, default=0)
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--map-id-list", type=int, nargs="+", default=[1])
    parser.add_argument("--num-agents", type=int, default=1)
    parser.add_argument("--use-depth-map", action="store_true")
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--replay-suffix", type=str, default="")
    parser.add_argument("--start-location", type=float, nargs=3, default=[0, 0, 0])
    parser.add_argument("--target-location", type=float, nargs=3, default=[5, 0, 5])
    parser.add_argument("--walk-speed", type=float, default=1)
    parser.add_argument("--map-dir", type=str, default=None)
    args = parser.parse_args()

    if args.map_dir is None:
        args.map_dir = os.path.expanduser("~/map_data")

    console.print(args, style="bold magenta")

    def get_picth_yaw(x, y, z):
        pitch = np.arctan2(y, (x**2 + z**2) ** 0.5) / np.pi * 180
        yaw = np.arctan2(x, z) / np.pi * 180
        return pitch, yaw

    #  currently same as simple navigation policy
    def my_policy(
        state, target_location=args.target_location, walk_speed=args.walk_speed
    ):
        agent_location = [state.position_x, state.position_y, state.position_z]
        direction = [v2 - v1 for v1, v2 in zip(agent_location, target_location)]
        yaw = get_picth_yaw(*direction)[1]
        turn_lr_delta = random.choice([2, 0, -2])
        look_ud_delta = random.choice([1, 0, -1])
        action = [yaw, walk_speed, turn_lr_delta, look_ud_delta, True]
        return action

    used_actions = [
        ActionVariable.WALK_DIR,
        ActionVariable.WALK_SPEED,
        ActionVariable.TURN_LR_DELTA,
        ActionVariable.LOOK_UD_DELTA,
        ActionVariable.PICKUP,
    ]

    game = Game(map_dir=args.map_dir, server_port=args.port)
    game.set_game_mode(args.game_mode)
    game.set_supply_heatmap_center([args.start_location[0], args.start_location[2]])
    game.set_supply_heatmap_radius(10)
    game.set_supply_indoor_richness(50)
    game.set_supply_outdoor_richness(10)
    game.set_supply_indoor_quantity_range(10, 50)
    game.set_supply_outdoor_quantity_range(1, 5)
    game.set_supply_spacing(6)
    game.set_episode_timeout(args.timeout)
    game.set_start_location(args.start_location)
    game.set_target_location(args.target_location)
    game.set_available_actions(used_actions)
    game.set_game_replay_suffix(args.replay_suffix)

    if args.use_depth_map:
        game.turn_on_depth_map()

    if args.record:
        game.turn_on_record()

    for agent_id in range(1, args.num_agents):
        game.add_agent(agent_name=f"agent_{agent_id}")

    game.init()

    for map_id in track(args.map_id_list, description="Running Maps ..."):
        game.set_map_id(map_id)
        if game.use_depth_map:
            w, h, f = [random.randint(10, 50) for _ in range(3)]
            game.set_depth_map_size(w, h, f)
        game.new_episode()

        console.print(game.get_game_config(), style="bold magenta")

        while not game.is_episode_finished():
            console.print(
                ">>>>>>>> TimeStep:", game.get_time_step(), style="bold magenta"
            )

            t = time.perf_counter()
            state_all = game.get_state_all()
            action_all = {
                agent_id: my_policy(state_all[agent_id]) for agent_id in state_all
            }
            game.make_action(action_all)
            dt = time.perf_counter() - t

            console.print(state_all, style="bold magenta")
            actions = {name: val for name, val in zip(used_actions, action_all[0])}
            console.print(actions, style="bold magenta")
            console.print("<<<<<<<<< StepRate:", round(1 / dt), style="bold magenta")

        print("episode ended ...")

    game.close()
