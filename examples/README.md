# Getting Started: basic examples and baselines

Here we provide some example python scripts, which may help you get familiar with the basic use of our environment quickly and possibly build your own AI training environment based on the provided game play interfaces. To run the scripts, here are some example shell commands for your reference:

## [basic.py](basic.py)

- Show the basic use of game playing interfaces
- All players are controlled by a naive random policy
- Multiple game configuration parameters (including `timeout`, `map_id`, `num_agents` etc.) can be set as you wish

```bash
# run the game in Navigation mode for one episode with 3 players and turn on depth map rendering
python basic.py \
--num-rounds 1 --num-agents 3 \
--record --replay-suffix basic_demo \
--use-depth-map
```

## [basic_track1_navigation.py](basic_track1_navigation.py)

- Show the basic use of game playing interfaces for track 1
- An agent is controlled by a simple navigation policy that tells the agent to walk towards the target position
- Multiple game configuration parameters (including `timeout`, `map_id`, `walk_speed` etc.) can be set as you wish

```bash
# run the game in Navigation mode for one episode with 1 player and turn on depth map rendering
python basic_track1_navigation.py \
--num-rounds 1 --num-agents 1 \
--record --replay-suffix simple_navigation \
--use-depth-map
```
