# Wilderness Scavenger: 3D Open-World FPS Game AI Challenge

This is a platform for intelligent agent learning based on a 3D open-world FPS game developed by Inspir.AI.

## Competition Overview

Focusing on learning intelligent agents in open-world games, this year we are hosting a new competition called *Wilderness Scavenger*. Featuring a battle royale-style 3D open-world gameplay experience and random PCG-based world generation, this new game will challenge participants to learn agents that can perform subtasks commonly seen in FPS games, such as navigation, scouting, and skirmishing. To win the competition, agents need to have a strong perception of complex 3D environments, then learn to exploit various environmental structures (such as terrain, buildings, and plants) by developing flexible strategies to gain advantages over other competitors. Despite of the difficulty of this goal, we hope that this new competition can serve a cornerstone of research in AI based game playing for open-world games.

## Features

- A light-weight 3D open-world FPS game developed with Unity3D game engine
- rendering-off game acceleration for fast training and evaluation
- large open world environment providing high freedom of game play strategies
- PCG-based map generation with randomly spawned buildings, plants and obstacles
- 100 scenario maps for generalized AI training

## Basic Structures

We developed this repository to provide a training and evaluation platform for the researchers interested in open-world FPS game AI. For getting started quickly, we summarize the basic structure of this repository as follows:

```bash
.
├── examples  # providing starter code examples and training baselines
│   ├── envs/...
│   ├── basic.py
│   ├── basic_track1_navigation.py
│   ├── basic_track2_supply_gather.py
│   ├── basic_track3_supply_battle.py
│   ├── baseline_track1_navigation.py
│   ├── baseline_track2_supply_gather.py
│   └── baseline_track3_supply_battle.py
├── inspirai_fps  # the game play API source code
│   ├── lib/...
│   ├── __init__.py
│   ├── gamecore.py
│   ├── raycast.py
│   ├── simple_command_pb2.py
│   ├── simple_command_pb2_grpc.py
│   └── utils.py
└── unity3d  # the game engine binaries and assets
    ├── UnityPlayer.so
    ├── fps.x86_64
    ├── fps_Data/...
    └── logs/...
```

- `unity3d`: the backend engine extracted from our game development project, containing all the game related assets, binaries and source codes.
- `inspirai_fps`: the python gameplay API for agent training and testing, providing the core [`Game`](inspirai_fps/gamecore.py) class and other useful tool classes and functions.
- `examples`: we provide basic starter codes for each game mode targeting each track of the challenge, and we also give out our implementation of some baseline solutions based on [`ray.rllib`](https://docs.ray.io/en/master/rllib/index.html) reinforcement learning framework.

## Supported Operating Systems

Currently, we only support **Linux**. We will update the support for Windows and MacOS soon.

## Installation (from source)

To use the game play API, you need to first install the package `inspirai_fps` by following the commands below:

```bash
git clone https://github.com/inspirai/wilderness-scavenger
cd wilderness-scavenger
pip install .
```

We recommend installing this package with python 3.8 (which is our development environment), so you may first create a virtual env using [`conda`](https://www.anaconda.com/) and finish installation:

```bash
$ conda create -n WildScav python=3.8
$ conda activate WildScav
(WildScav) $ pip install .
```

## Installation (from PyPI)

Alternatively, you can install the package from PyPI directly. But note that this will only install the gameplay API `inspirai_fps`, not the backend engine. So you still need to manually download the engine binaries and assets (`unity3d`) from our repository.

```bash
pip install inspirai-fps
```

## Loading Game Engine

To successfully run the game, you need to make sure the game engine folder `unity3d` is downloaded along with the repository and set the `engine_dir` parameter of the `Game` init function correctly. For example, here is a code snippet in the script `example/basic.py`:

```python
parser.add_argument("--engine-dir", type=str, default="../unity3d")
...
game = Game(..., engine_dir=args.engine_dir, ...)
```

## Loading Map Data

To get access to some features like realtime depth map computation or randomized player spawning, you need to load the map data and load them into the `Game`. After this, once you turn on the depth map rendering, the game server will automatically compute a depth map viewing from the player's first person perspective at each time step.

1. Download world meshes from [data_meshes](https://drive.google.com/file/d/1SY43c5Gg8x-bxzqIazxuV8vOKCAh4LI2/view?usp=sharing) and the valid location lists from [data_locations](https://drive.google.com/file/d/1g_oC9hC7mrlKeDUtyU-y-izlblQRIp-D/view?usp=sharing)
2. Unzip all mesh (`xxx.obj`) and location (`xxx.json`) files to the same folder (e.g. `'<WORKDIR>/map_data'`)
3. Set `map_dir` parameter of the `Game` init function accordingly
4. Set the `map_id` as you like
5. Turn on the function of depth map computation
6. Turn on random start location for player initialization

See following code snippet in the script `examples/basic.py` for example:

```python
parser.add_argument("-I", "--map-id", type=int, default=1)
parser.add_argument("--use-depth-map", action="store_true")
parser.add_argument("--random-start-location", action="store_true")
parser.add_argument("--map-dir", type=str, default="../data")
...
game = Game(map_dir=args.map_dir, ...)
game.set_map_id(args.map_id)  # this will load the locations and mesh of the map with the given map id
...
if args.use_depth_map:
    game.turn_on_depth_map()

if args.random_start_location:
    for agent_id in range(args.num_agents):
        game.random_start_location(agent_id)  # this will randomly spawn the player at a valid location
```
