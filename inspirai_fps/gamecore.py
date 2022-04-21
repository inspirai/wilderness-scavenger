import os
import grpc
import time
import random
import numpy as np
import subprocess
from queue import Queue
from concurrent import futures
from typing import List, Tuple, Any

from google.protobuf.json_format import MessageToDict

from inspirai_fps import simple_command_pb2
from inspirai_fps import simple_command_pb2_grpc
from inspirai_fps.raycast import RaycastManager
from inspirai_fps.utils import get_orientation, get_position, set_vector3d, set_GM_command, load_json, vector3d_to_list, get_distance


__all__ = [
    "ActionVariable",
    "Game",
]

print_server_log = False
SPEED_UP_FACTOR = 10
TRIGGER_DISTANCE = 1


class QueueServer(simple_command_pb2_grpc.CommanderServicer):
    def __init__(self, request_queue, reply_queue) -> None:
        super().__init__()
        self.request_queue = request_queue
        self.reply_queue = reply_queue

    def Request_S2A_UpdateGame(self, request, context):
        self.request_queue.put(request)

        if print_server_log:
            print("Put request into queue ...")
            print(request)

        reply = self.reply_queue.get()

        if print_server_log:
            print("Get reply from queue ...")
            print(reply)

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
    HIT_BY_ENEMY = "hit_by_enemy"
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

    def __repr__(self) -> str:
        x = self.position_x
        y = self.position_y
        z = self.position_z
        return f"Supply_Pos(x={x:.2f},y={y:.2f},z={z:.2f})_Num({self.quantity})"


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
        self.move_dir_x = enemy_info.move_dir.x
        self.move_dir_y = enemy_info.move_dir.y
        self.move_dir_z = enemy_info.move_dir.z
        self.move_speed = enemy_info.move_speed

    def __repr__(self) -> str:
        x = self.position_x
        y = self.position_y
        z = self.position_z
        return f"Enemy_Pos(x={x:.2f},y={y:.2f},z={z:.2f})_Speed({self.move_speed})"


class EnemyStateRough:
    """show simplified information of the enemy, including:
    - dir_vec: a 2d unit vector pointing from the agent's location to the enemy's location
    """

    def __init__(self, enemy_info: simple_command_pb2.EnemyInfo, obs_data: simple_command_pb2.Observation) -> None:
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
    - time_step `int`: how many steps the game has run for
    - game_state `int`: current state of the game (update | over)
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

    def __init__(self, obs_data, time_step, game_state, ray_tracer, use_depth_map=False, supply_visible_distance=np.Inf) -> None:

        self.ray_tracer = ray_tracer
        self.supply_visible_distance = supply_visible_distance

        self.time_step = time_step  # TODO: remove this
        self.game_state = game_state  # TODO: remove this
        self.position_x = obs_data.location.x
        self.position_y = obs_data.location.y
        self.position_z = obs_data.location.z
        self.move_dir_x = obs_data.move_dir.x
        self.move_dir_y = obs_data.move_dir.y
        self.move_dir_z = obs_data.move_dir.z
        self.move_speed = obs_data.move_speed
        self.pitch = obs_data.pitch
        self.yaw = obs_data.yaw
        self.health = obs_data.hp
        self.weapon_ammo = obs_data.num_gun_ammo
        self.spare_ammo = obs_data.num_pack_ammo
        self.on_ground = obs_data.on_ground
        self.is_attack = obs_data.is_fire
        self.is_reload = obs_data.is_reload
        self.hit_enemy = obs_data.hit_enemy
        self.hit_by_enemy = obs_data.hit_by_enemy
        self.num_supply = obs_data.num_supply
        self.is_waiting_respawn = obs_data.is_waiting_respawn
        self.is_invincible = obs_data.is_invincible

        self.depth_map = None
        if use_depth_map:
            pos = get_position(self)
            dir = get_orientation(self)
            self.depth_map = self.ray_tracer.get_depth(pos, dir)[0]

        self.supply_states = [SupplyState(s) for s in filter(self.is_supply_visible, obs_data.supply_info_list)]
        self.enemy_states = [EnemyStateDetailed(e) for e in filter(self.is_enemy_visible, obs_data.enemy_info_list)]

    def __repr__(self) -> str:
        x = self.position_x
        y = self.position_y
        z = self.position_z
        return f"GameState[ts={self.time_step}][x={x:.2f},y={y:.2f},z={z:.2f}][supply={self.num_supply}][gun_ammo={self.weapon_ammo}][pack_ammo={self.spare_ammo}]"

    def is_enemy_visible(self, enemy_info: simple_command_pb2.EnemyInfo):
        view_angle = [90 + 20, (90 + 20) / 16 * 9]
        body_param = [0.45 * 2, 1.78]  # body size (width, height)
        camera_height = 1.5

        agent_team_id = [0, 1]
        agent_position = [
            self.position_x,
            self.position_z,
            self.position_y,
            enemy_info.location.x,
            enemy_info.location.z,
            enemy_info.location.y,
        ]
        camera_location = [
            self.position_x,
            self.position_z + camera_height,
            self.position_y,
            enemy_info.location.x,
            enemy_info.location.z + camera_height,
            enemy_info.location.y,
        ]
        camerarotation = [
            0,
            self.pitch,
            self.yaw,
            0,
            0,  # default 0 -> of no use
            0,  # default 0 -> of no use
        ]

        is_visible = self.ray_tracer.agent_is_visible(body_param, view_angle, agent_team_id, agent_position, camera_location, camerarotation)

        return is_visible[0][1]

    def is_supply_visible(self, supply_info: simple_command_pb2.SupplyInfo):
        supply_pos = vector3d_to_list(supply_info.supply_location)
        self_pos = get_position(self)
        distance = get_distance(self_pos, supply_pos)
        return distance <= self.supply_visible_distance


class Game:
    MODE_NAVIGATION = simple_command_pb2.GameModeType.NAVIGATION_MODE
    MODE_SUP_GATHER = simple_command_pb2.GameModeType.SUP_GATHER_MODE
    MODE_SUP_BATTLE = simple_command_pb2.GameModeType.SUP_BATTLE_MODE

    def __init__(self, map_dir="../map_data", engine_dir="../unity3d", server_port=50051):
        self.map_dir = map_dir
        self.valid_locations = []
        self.unity_exec_path = os.path.join(engine_dir, "fps.x86_64")
        self.unity_log_name = f"game_server_port={server_port}"
        self.unity_log_dir = os.path.join(engine_dir, "logs")
        self.server_ip = "127.0.0.1"
        self.server_port = server_port
        self.available_actions = []
        self.use_depth_map = False
        self.GM = self.__get_default_GM()
        self.__set_time_scale(SPEED_UP_FACTOR)
        self.__set_trigger_range(TRIGGER_DISTANCE)

    def __get_default_GM(self):
        gm_command = simple_command_pb2.GMCommand()

        # Common settings
        gm_command.timeout = 60
        gm_command.game_mode = self.MODE_NAVIGATION
        gm_command.time_scale = 1
        gm_command.map_id = 1
        gm_command.random_seed = 0
        gm_command.num_agents = 1
        gm_command.is_record = False
        gm_command.replay_suffix = ""

        # ModeNavigation settings
        set_vector3d(gm_command.target_location, [1, 0, 1])
        gm_command.trigger_range = 1

        # ModeSupplyGather settings
        set_vector3d(gm_command.supply_heatmap_center, [0, 0, 0])
        gm_command.supply_heatmap_radius = 50
        gm_command.supply_create_percent = 50
        gm_command.supply_house_create_percent = 50
        gm_command.supply_grid_length = 3
        gm_command.supply_random_min = 1
        gm_command.supply_random_max = 1
        gm_command.supply_house_random_min = 10
        gm_command.supply_house_random_max = 10

        # ModeSupplyBattle settings
        gm_command.respawn_time = 10

        # Default player settings
        agent = gm_command.agent_setups.add()
        agent.id = 0
        agent.hp = 100
        agent.num_pack_ammo = 60
        agent.gun_capacity = 15
        agent.attack_power = 20
        agent.agent_name = "InspirAI"
        set_vector3d(agent.start_location, [0, 1, 0])

        return gm_command

    def set_game_config(self, config_path: str):
        """Experimental: Set game config from a yaml file"""
        game_config = load_json(config_path)
        self.GM.Clear()
        set_GM_command(self.GM, game_config)

    def get_game_config(self):
        return MessageToDict(self.GM)

    def get_agent_name(self, agent_id):
        for agent in self.GM.agent_setups:
            if agent.id == agent_id:
                return agent.agent_name

    def set_episode_timeout(self, timeout: int):
        assert isinstance(timeout, int) and timeout > 0
        self.GM.timeout = timeout

    def set_game_mode(self, game_mode: int):
        assert game_mode in [0, 1, 2]
        self.GM.game_mode = game_mode

    def __set_time_scale(self, time_scale: int):
        assert isinstance(time_scale, int) and time_scale >= 1
        self.GM.time_scale = time_scale

    def set_map_id(self, map_id: int):
        assert isinstance(map_id, int) and 1 <= map_id <= 100
        self.valid_locations = load_json(f"{self.map_dir}/{map_id:03d}.json")
        self.GM.map_id = map_id

    def get_valid_locations(self):
        return self.valid_locations

    def set_random_seed(self, random_seed: int):
        assert isinstance(random_seed, int)
        self.GM.random_seed = random_seed

    def set_start_location(self, loc: List[float], agent_id: int = 0):
        assert isinstance(agent_id, int) and 0 <= agent_id < len(self.GM.agent_setups)
        assert isinstance(loc, list) and len(loc) == 3
        agent = self.GM.agent_setups[agent_id]
        set_vector3d(agent.start_location, loc)

    def get_start_location(self, agent_id: int = 0):
        for agent in self.GM.agent_setups:
            if agent.id == agent_id:
                loc = agent.start_location
                return [loc.x, loc.y, loc.z]

    def random_start_location(self, agent_id: int = 0):
        assert isinstance(agent_id, int) and 0 <= agent_id < len(self.GM.agent_setups)
        agent = self.GM.agent_setups[agent_id]
        loc = random.choice(self.valid_locations)
        set_vector3d(agent.start_location, loc)

    def set_target_location(self, loc: List[float]):
        assert isinstance(loc, list) and len(loc) == 3
        set_vector3d(self.GM.target_location, loc)

    def get_target_location(self):
        loc = self.GM.target_location
        return loc.x, loc.y, loc.z

    def set_available_actions(self, actions: List[str]):
        assert isinstance(actions, list) and len(actions) >= 1
        self.available_actions = actions
        self.action_idx_map = {key: i for i, key in enumerate(self.available_actions)}

    def __set_trigger_range(self, trigger_range: float):
        assert isinstance(trigger_range, (float, int)) and trigger_range > 0.5
        self.GM.trigger_range = float(trigger_range)

    def set_game_replay_suffix(self, replay_suffix: str):
        assert isinstance(replay_suffix, str)
        self.GM.replay_suffix = replay_suffix

    def set_supply_heatmap_center(self, loc: List[float]):
        """loc: a list of two numbers that represent the x and z values of the center location"""
        assert isinstance(loc, list) and len(loc) == 2
        assert -150 <= loc[0] <= 150
        assert -150 <= loc[1] <= 150
        center = [loc[0], 0, loc[1]]
        set_vector3d(self.GM.supply_heatmap_center, center)

    def set_supply_heatmap_radius(self, radius: int):
        assert isinstance(radius, int) and 1 <= radius <= 200
        self.GM.supply_heatmap_radius = radius

    def set_supply_outdoor_richness(self, richness: int):
        assert isinstance(richness, int) and 0 <= richness <= 50
        self.GM.supply_create_percent = richness

    def set_supply_indoor_richness(self, richness: int):
        assert isinstance(richness, int) and 0 <= richness <= 100
        self.GM.supply_house_create_percent = richness

    def set_supply_spacing(self, spacing: int):
        assert isinstance(spacing, int) and spacing >= 1
        self.GM.supply_grid_length = spacing

    def set_supply_outdoor_quantity_range(self, qmin=1, qmax=1):
        assert isinstance(qmin, int) and isinstance(qmax, int) and 1 <= qmin <= qmax
        self.GM.supply_random_min = qmin
        self.GM.supply_random_max = qmax

    def set_supply_indoor_quantity_range(self, qmin=1, qmax=1):
        assert isinstance(qmin, int) and isinstance(qmax, int) and 1 <= qmin <= qmax
        self.GM.supply_house_random_min = qmin
        self.GM.supply_house_random_max = qmax

    def add_supply_refresh(self, refresh_time: int, heatmap_radius: int, heatmap_center: List[float], indoor_richness: int, outdoor_richness: int):
        assert isinstance(refresh_time, int) and refresh_time >= 1
        assert isinstance(heatmap_radius, int) and 1 <= heatmap_radius <= 200
        assert isinstance(heatmap_center, list) and len(heatmap_center) == 2
        assert -150 <= heatmap_center[0] <= 150
        assert -150 <= heatmap_center[1] <= 150
        assert isinstance(indoor_richness, int) and 0 <= indoor_richness <= 100
        assert isinstance(outdoor_richness, int) and 0 <= outdoor_richness <= 50

        refresh = self.GM.supply_refresh_datas.add()
        center = [heatmap_center[0], 0, heatmap_center[1]]
        set_vector3d(refresh.supply_heatmap_center, center)
        refresh.supply_heatmap_radius = heatmap_radius
        refresh.supply_refresh_time = refresh_time
        refresh.supply_create_percent = outdoor_richness
        refresh.supply_house_create_percent = indoor_richness

    def add_agent(self, health=100, num_pack_ammo=60, num_clip_ammo=15, attack=20, start_location=[0, 0, 0], agent_name=None):
        assert isinstance(start_location, list) and len(start_location) == 3
        assert isinstance(health, int) and 1 <= health <= 100
        assert isinstance(num_clip_ammo, int) and num_clip_ammo >= 0
        assert isinstance(num_pack_ammo, int) and num_pack_ammo >= 0
        assert isinstance(attack, int) and attack >= 1

        agent = self.GM.agent_setups.add()
        agent.id = self.GM.num_agents
        agent_name = agent_name or f"Agent-{agent.id}"
        assert isinstance(agent_name, str) and len(agent_name) > 0
        agent.agent_name = agent_name

        agent.hp = health
        agent.num_pack_ammo = num_pack_ammo
        agent.gun_capacity = num_clip_ammo
        agent.attack_power = attack

        set_vector3d(agent.start_location, start_location)
        self.GM.num_agents += 1

    def turn_on_record(self):
        self.GM.is_record = True

    def turn_off_record(self):
        self.GM.is_record = False

    def turn_on_depth_map(self):
        self.use_depth_map = True

    def turn_off_depth_map(self):
        self.use_depth_map = False

    def get_depth_map_height(self):
        """deprecated: use get_depth_map_size instead"""
        return RaycastManager.HEIGHT

    def get_depth_map_width(self):
        """deprecated: use get_depth_map_size instead"""
        return RaycastManager.WIDTH

    def get_depth_limit(self):
        """deprecated: use get_depth_map_size instead"""
        return RaycastManager.DEPTH

    def get_depth_map_size(self):
        return RaycastManager.WIDTH, RaycastManager.HEIGHT, RaycastManager.DEPTH

    def get_frame_count(self):
        return self.latest_request.time_step

    def get_target_reach_distance(self):
        return self.GM.trigger_range

    def init(self):
        assert len(self.available_actions) > 0
        self.request_queue = Queue()
        self.reply_queue = Queue()

        # initialize message queue server
        self.server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
        simple_command_pb2_grpc.add_CommanderServicer_to_server(QueueServer(self.request_queue, self.reply_queue), self.server)
        self.server.add_insecure_port(f"[::]:{self.server_port}")
        self.server.start()
        print("Server started ...")

        # start unity3d game engine
        os.system(f"chmod +x {self.unity_exec_path}")

        os.makedirs(self.unity_log_dir, exist_ok=True)
        log_path = os.path.join(self.unity_log_dir, f"{self.unity_log_name}.log")
        with open(log_path, "w") as f:
            cmd = f"{self.unity_exec_path} -IP:{self.server_ip} -PORT:{self.server_port}"
            subprocess.Popen(cmd, shell=True, stdout=f, bufsize=-1)
        print("Unity3D started ...")

        # waiting for unity3d to send the first request
        self.request_queue.get()  # the first request is only used to activate the server
        print("Unity3D connected ...")

    def get_state(self, agent_id=0):
        for obs_data in self.latest_request.agent_obs_list:
            if obs_data.id == agent_id:
                return AgentState(obs_data, self.latest_request.time_step, self.latest_request.game_state, self.ray_tracer, self.use_depth_map)

    def get_state_all(self):
        state_dict = {}
        time_step = self.latest_request.time_step
        game_state = self.latest_request.game_state

        for obs_data in self.latest_request.agent_obs_list:
            agent_id = obs_data.id
            state_dict[agent_id] = AgentState(obs_data, time_step, game_state, self.ray_tracer, self.use_depth_map)
        return state_dict

    def make_action(self, action_cmd_dict):
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
            for action_name, idx in self.action_idx_map.items():
                setattr(agent_cmd, action_name, action[idx])

        self.reply_queue.put(reply)
        self.latest_reply = reply
        self.latest_request = self.request_queue.get()

    def is_episode_finished(self):
        return self.latest_request.game_state == simple_command_pb2.GameState.over

    def new_episode(self):
        reply = simple_command_pb2.A2S_Reply_Data()
        reply.game_state = simple_command_pb2.GameState.reset
        reply.gm_cmd.CopyFrom(self.GM)
        self.reply_queue.put(reply)
        self.lastest_reply = reply
        self.latest_request = self.request_queue.get()
        print("Started new episode ...")

        mesh_file_path = f"{self.map_dir}/{self.GM.map_id:03d}.obj"
        self.ray_tracer = RaycastManager(mesh_file_path)
        print(f"Map {self.GM.map_id:03d} loaded ...")

    def close(self):
        reply = simple_command_pb2.A2S_Reply_Data()
        reply.game_state = simple_command_pb2.GameState.close
        self.reply_queue.put(reply)
        time.sleep(1)
        self.server.stop(0)
        print("Server stopped ...")

        cmd = "ps -ef | grep fps.x86_64 | grep PORT:" + str(self.server_port) + " | awk '{print $2}' | xargs kill -9"
        os.system(cmd)
        print("Unity3D stopped ...")


if __name__ == "__main__":
    import argparse
    from rich.console import Console
    from rich.progress import track

    console = Console()

    parser = argparse.ArgumentParser()
    parser.add_argument("-P", "--port", type=int, default=50051)
    parser.add_argument("-T", "--timeout", type=int, default=10)
    parser.add_argument("-M", "--game-mode", type=int, default=0)
    parser.add_argument("-R", "--time-scale", type=int, default=10)
    parser.add_argument("-I", "--map-id", type=int, default=1)
    parser.add_argument("-S", "--random-seed", type=int, default=0)
    parser.add_argument("-N", "--num-rounds", type=int, default=1)
    parser.add_argument("-TR", "--trigger-range", type=float, default=1)
    parser.add_argument("--num-agents", type=int, default=1)
    parser.add_argument("--use-depth-map", action="store_true")
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--replay-suffix", type=str, default="")
    parser.add_argument("--start-location", type=float, nargs=3, default=[0, 0, 0])
    parser.add_argument("--target-location", type=float, nargs=3, default=[5, 0, 5])
    parser.add_argument("--walk-speed", type=float, default=1)
    args = parser.parse_args()

    def get_picth_yaw(x, y, z):
        pitch = np.arctan2(y, (x**2 + z**2) ** 0.5) / np.pi * 180
        yaw = np.arctan2(x, z) / np.pi * 180
        return pitch, yaw

    def my_policy(state):  #  currently same as simple navigation policy
        self_pos = [state.position_x, state.position_y, state.position_z]
        target_pos = args.target_location
        direction = [v2 - v1 for v1, v2 in zip(self_pos, target_pos)]
        yaw = get_picth_yaw(*direction)[1]
        turn_lr_delta = random.choice([1, 0, -1])
        action = [yaw, args.walk_speed, turn_lr_delta, True]
        return action

    used_actions = [
        ActionVariable.WALK_DIR,
        ActionVariable.WALK_SPEED,
        ActionVariable.TURN_LR_DELTA,
        ActionVariable.PICKUP,
    ]

    game = Game(server_port=args.port)
    game.set_game_mode(args.game_mode)
    game.set_supply_heatmap_center([args.start_location[0], args.start_location[2]])
    game.set_supply_heatmap_radius(50)
    game.set_supply_indoor_richness(80)
    game.set_supply_outdoor_richness(20)
    game.set_supply_indoor_quantity_range(10, 50)
    game.set_supply_outdoor_quantity_range(1, 5)
    game.set_supply_spacing(3)
    game.set_episode_timeout(args.timeout)
    game.set_start_location(args.start_location)
    game.set_target_location(args.target_location)
    game.__set_trigger_range(args.trigger_range)
    game.set_available_actions(used_actions)
    if args.use_depth_map:
        game.turn_on_depth_map()
    if args.record:
        game.turn_on_record()

    for agent_id in range(1, args.num_agents):
        start_loc = [random.randint(-10, 10), 3, random.randint(-10, 10)]
        game.add_agent(start_location=start_loc)

    game.set_map_id(args.map_id)
    game.init()

    for ep in track(range(args.num_rounds), description="Running Episodes ..."):
        game.set_game_replay_suffix(f"{args.replay_suffix}_episode_{ep}")
        game.new_episode()

        while not game.is_episode_finished():
            t = time.time()
            state_all = game.get_state_all()
            action_all = {agent_id: my_policy(state_all[agent_id]) for agent_id in state_all}
            game.make_action(action_all)
            dt = time.time() - t

            agent_id = 0
            state = state_all[agent_id]
            step_info = {
                "Episode": ep,
                "GameState": state.game_state,
                "TimeStep": state.time_step,
                "AgentID": agent_id,
                "Location": get_position(state),
                "Action": {name: val for name, val in zip(used_actions, action_all[agent_id])},
                "#SupplyInfo": len(state.supply_states),
                "#EnemyInfo": len(state.enemy_states),
                "StepRate": round(1 / dt),
                # "DepthMap": state.depth_map.shape
            }
            console.print(step_info, style="bold magenta")

        print("episode ended ...")

    game.close()
