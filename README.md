# Wilderness Scavenger: 3D Open-World FPS Game AI Challenge

This is a platform for intelligent agent learning based on a 3D open-world FPS game developed by Inspir.AI.

## Competition Overview

Focusing on open-world FPS game AI, this year we are hosting a new competition called 3D Open World FPS AI Challenge. Featuring a battle royale-style 3D open environment and random PCG-based world generation, this new game will challenge AI agents to some of the most important skills in FPS games, such as navigation, scouting, and skirmishing. To win the competition, agents need to have a strong perception of complex 3D environments, then learn to exploit various environmental structures (such as terrain, buildings, and plants) and develop highly flexible strategies to gain an advantage over competitors. Although the problem is difficult, we hope that this new competition will become a cornerstone of future AI research in open-world FPS games.

## Features


- A light-weight 3D open-world FPS game developed with Unity3D game engine
- rendering-off game acceleration for fast training and evaluation
- large open world environment providing high freedom of the player's exploration
- PCG-based terrain generation with randomly spawned buildings, plants and obstacles
- 100 scenario maps for generalized AI training

## Basic Structures

We developed this repository to provide a training and evaluation platform for the researchers interested in open-world FPS game AI. For getting started quickly, we summarize the basic structure of this repository as follows:

```bash
.
├── data                               # storing map data in the format [map_id].json [map_id].obj
│   ├── 001.json
│   └── 001.obj
├── examples                           # providing starter code examples and training baselines
│   ├── basic.py
│   ├── basic_track1_navigation.py
│   ├── basic_track2_supply_gather.py
│   ├── basic_track3_supply_battle.py
│   ├── baseline_track1_normal.py
│   ├── baseline_track1_hard.py
│   ├── baseline_track2_normal.py
│   ├── baseline_track2_hard.py
│   ├── baseline_track3_normal.py
│   └── baseline_track3_hard.py
├── inspirai_fps                       # the game play API source code
│   ├── lib/...
│   ├── __init__.py
│   ├── gamecore.py
│   ├── raycast.py
│   ├── simple_command_pb2.py
│   ├── simple_command_pb2_grpc.py
│   └── utils.py
└── unity3d                            # the game engine binaries and assets
    ├── UnityPlayer.so
    ├── fps.x86_64
    ├── fps_Data/...
    └── logs/...
```

- `unity3d`: the engine backend extracted from our game development project, containing all the game related assets, binaries and source codes.
- `inspirai_fps`: the game playing python APIs for agent learning, providing the core [`Game`](inspirai_fps/gamecore.py) class and other useful tool classes and functions.
- `data`: storing all map related data files and they are used for calculating some observation attributes in the [`AgentState`](inspirai_fps/gamecore.py).
- `examples`: we provide basic starter codes for each game mode targeting each track of the challenge, and we also give out our implementation of some baseline solutions based on [`ray.rllib`](https://docs.ray.io/en/master/rllib/index.html) reinforcement learning framework.

## Installation

To use the game play API, you need to first install the package `inspirai_fps` by following the commands below:

```bash
$ git clone https://github.com/inspirai/wilderness-scavenger
$ cd wilderness-scavenger
$ pip install .
```

We recommend installing this package with python3.8 (which is our development environment), so you may first create a virtual env using [`conda`](https://www.anaconda.com/) and finish installation:

```bash
$ conda create -n InspiraiFPS python=3.8
$ conda activate InspiraiFPS
(InspiraiFPS) $ pip install .
```

## Loading Game Engine

To successfully run the game, you need to make sure the game engine folder `unity3d` is downloaded along with the repository and set the `engine_dir` parameter of the `Game` init function correctly. For example, here is a code snippet in the script `example/basic.py`:

```python
parser.add_argument("--engine-dir", type=str, default="../unity3d")
...
game = Game(..., engine_dir=args.engine_dir, ...)
```

## Loading Map Data

To utilize some advanced observation features like depth map computation, we need to load the mesh data into the game API and then it will automatically compute a depth map viewing from the player's first person perspective at each time step.

1. Download the map data from this link [map_data]()
2. Set the correct map data root `map_dir` to your downloaded data directory
3. Set the desired `map_id` and turn on the function of depth map computing 

See following code snippet in the script `examples/basic.py` for example:

```python
parser.add_argument("-I", "--map-id", type=int, default=1)
parser.add_argument("--use-depth-map", action="store_true")
parser.add_argument("--map-dir", type=str, default="../data")
...
game = Game(map_dir=args.map_dir, ...)
game.set_map_id(args.map_id)
...
if args.use_depth_map:
    game.turn_on_depth_map()
```
