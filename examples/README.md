# Getting started with inspirai_fps API

Here we provide some example python scripts, which may help you get familiar with the basic use of our environment quickly and possibly build your own AI training environment based on the provided game play interfaces. To run the scripts, here are some example shell commands for your reference:

## [Basic.py](basic.py)

- show the basic use of game playing interfaces
- all players are manipulated by the same artificial control policy, which implements a very simple navigation function
- multiple game configuration parameters (including `timeout`, `map_id`, `num_agents` etc.) can be set as you wish

```bash
# run the environment in multiplayer mode for one game with 3 players and turn on depth map calculation
$ python basic.py --num-rounds 1 -game-mode 2 --num-agents 3 --record --replay-suffix fps_pack --use-depth-map
```
