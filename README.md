# Wilderness Scavenger: 3D Open-World FPS Game AI Challenge

This is a platform for intelligent agent learning based on a 3D open-world FPS game developed by Inspir.AI.

## Competition Overview

With a focus on learning intelligent agents in open-world games, this year we are hosting a new contest called *Wilderness Scavenger*. In this new game, which features a Battle Royale-style 3D open-world gameplay experience and a random PCG-based world generation, participants must learn agents that can perform subtasks common to FPS games, such as navigation, scouting, and skirmishing. To win the competition, agents must have strong perception of complex 3D environments and then learn to exploit various environmental structures (such as terrain, buildings, and plants) by developing flexible strategies to gain advantages over other competitors. Despite the difficulty of this goal, we hope that this new competition can serve as a cornerstone of research in AI-based gaming for open-world games.

## Features

- A light-weight 3D open-world FPS game developed with Unity3D game engine
- Rendering-off game acceleration for fast training and evaluation
- Large open world environment providing high freedom of agent behaviors
- Highly customizable game configuration with random supply distribution and dynamic refresh
- PCG-based map generation with randomly spawned buildings, plants and obstacles (100 training maps)
- Interactive replay tool for game record visualization

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
│   ├── raycast_manager.py
│   ├── simple_command_pb2.py
│   ├── simple_command_pb2_grpc.py
│   └── utils.py
└── unity3d  # the engine backend (default Linux)
    ├── UnityPlayer.so
    ├── fps.x86_64
    ├── fps_Data/...
    └── logs/...
```

- `unity3d`: the (default Linux) backend engine extracted from our game development project, containing all the game related assets, binaries and source codes.
- `inspirai_fps`: the python gameplay API for agent training and testing, providing the core [`Game`](inspirai_fps/gamecore.py) class and other useful tool classes and functions.
- `examples`: we provide basic starter codes for each game mode targeting each track of the challenge, and we also give out our implementation of some baseline solutions based on [`ray.rllib`](https://docs.ray.io/en/master/rllib/index.html) reinforcement learning framework.

## Supported Platforms

We support the multiple platforms with different engine backends, including:

<!-- - Windows: download the engine [here](https://drive.google.com/file/d/1CEpiFPpx5NsWgqL8yzaZQX9fGuyPUKDy/view?usp=sharing)
- MacOS: download the engine [here](https://drive.google.com/file/d/1hgQa5OPve4QCBczLEGeOis2HhHbxI68m/view?usp=sharing) -->
- Linux: download the engine [here]()
- Windows: will be updated soon
- MacOS: will be updated soon

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

**Note: this may not be maintained in time.**

Alternatively, you can install the package from PyPI directly. But note that this will only install the gameplay API `inspirai_fps`, not the backend engine. So you still need to manually download the engine binaries and assets (`unity3d`) from our repository.

```bash
pip install inspirai-fps
```

## Loading Engine Backend

To successfully run the game, you need to make sure the game engine backend for your platform is downloaded and set the `engine_dir` parameter of the `Game` init function correctly. For example, here is a code snippet in the script `example/basic.py`:

```python
from inspirai_fps import Game, ActionVariable
...
parser.add_argument("--engine-dir", type=str, default="../unity3d")
...
game = Game(..., engine_dir=args.engine_dir, ...)
```

## Loading Map Data

To get access to some features like realtime depth map computation or randomized player spawning, you need to load the map data and load them into the `Game`. After this, once you turn on the depth map rendering, the game server will automatically compute a depth map viewing from the player's first person perspective at each time step.

1. Download map data [here](https://drive.google.com/file/d/1QGrKfnVZ2Z7f2JPjLbYAQy5Pv6y8vz3p/view?usp=sharing) and decompress the downloaded file to your preferred directory (e.g., `<WORKDIR>/map_data`).
2. Set `map_dir` parameter of the `Game` initializer accordingly
3. Set the `map_id` as you like
4. Turn on the function of depth map computation
5. Turn on random start location to spawn agents at random places

Read the following code snippet in the script `examples/basic.py` as an example:

```python
from inspirai_fps import Game, ActionVariable
...
parser.add_argument("--map-id", type=int, default=1)
parser.add_argument("--use-depth-map", action="store_true")
parser.add_argument("--random-start-location", action="store_true")
parser.add_argument("--map-dir", type=str, default="../data")
...
game = Game(map_dir=args.map_dir, ...)
game.set_map_id(args.map_id)  # this will load the valid locations of the specified map
...
if args.use_depth_map:
    game.turn_on_depth_map()
...
if args.random_start_location:
    for agent_id in range(args.num_agents):
        game.random_start_location(agent_id)  # this will randomly spawn the player at a valid location
...
game.new_episode()  # start a new episode, this will load the mesh of the specified map
```

## Gameplay Visualization

We have also developed a replay visualization tool based on the Unity3D game engine. It is similar to the spectator mode common in multiplayer FPS games, which allows users to interactively follow the gameplay. Users can view an agent's action from different perspectives and also switch between multiple agents or different viewing modes (e.g., first person, third person, free) to see the entire game in a more immersive way. Participants can download the tool for their specific platforms here:

- Windows: download the replay tool [here](https://drive.google.com/file/d/1YIEGnjKaH_KzycwJK5WKEGMVn8dls7dR/view?usp=sharing)
- MacOS: download the replay tool [here](https://drive.google.com/file/d/1QKfMmF_4FZc2hJ2cEzD6psv6jr21rC_L/view?usp=sharing)

To use this tool, follow the instruction below:

- Decompress the downloaded file to anywhere you prefer.
- Turn on recording function with `game.turn_on_record()`. One record file will be saved at the end of each episode.

Find the replay files under the engine directory according to your platform:

- Linux: `<engine_dir>/fps_Data/StreamingAssets/Replay`
- Windows: `<engine_dir>\FPSGameUnity_Data\StreamingAssets\Replay`
- MacOS: `<engine_dir>/Contents/Resources/Data/StreamingAssets/Replay`

Copy replay files you want to the replay tool directory according to your platform and start the replay tool.

For Windows users:

- Copy the replay file (e.g. `xxx.bin`) into `<replayer_dir>/FPSGameUnity_Data/StreamingAssets/Replay`
- Run `FPSGameUnity.exe` to start the application.

For MacOS users:

- Copy the replay file (e.g. `xxx.bin`) into `<replayer_dir>/Contents/Resources/Data/StreamingAssets/Replay`
- Run `fps.app` to start the application.

In the replay tool, you can:

- Select the record you want to watch from the drop-down menu and click **PLAY** to start playing the record.
- During the replay, users can make the following operations
  - Press **Tab**: pause or resume
  - Press **E**: switch observation mode (between first person, third person, free)
  - Press **Q**: switch between multiple agents
  - Press **ECS**: stop replay and return to the main menu
