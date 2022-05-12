# Getting Started: basic examples for quick start

Here we provide some example python scripts, which may help you get familiar with the basic use of our environment quickly and possibly build your own AI training environment based on the provided game play interfaces. To run the scripts, here are some example shell commands for your reference:

## [basic.py](basic.py)

- Show the basic use of game playing interfaces
- All players are controlled by a naive simple policy
- Multiple game configuration parameters (including `timeout`, `map_id`, `num_agents` etc.) can be set as you wish

```bash
# run the game in Navigation mode for one episode with 3 players and turn on depth map rendering
python basic.py \
--num-episodes 1 --num-agents 3 \
--record --replay-suffix basic_demo \
--use-depth-map --game-mode 0  
```

## [basic_track1_navigation.py](basic_track1_navigation.py)

- Show the basic use of game playing interfaces for track 1
- An agent is controlled by a simple navigation policy that tells the agent to walk towards the target position
- Multiple game configuration parameters (including `timeout`, `map_id`, `walk_speed` etc.) can be set as you wish

```bash
# run the game in Navigation mode for one episode with 1 player and turn on depth map rendering
python basic_track1_navigation.py \
--num-episodes 1 \
--record --replay-suffix simple_navigation \
--use-depth-map
```

## [basic_track2_supply_gather.py](basic_track2_supply_gather.py)

- Show the basic use of game playing interfaces for track 2
- An agent is controlled by a simple gather policy that tells the agent to act randomly
- Multiple game configuration parameters (including `timeout`, `map_id`, `walk_speed` etc.) can be set as you wish

```bash
# run the game in Supply_gather mode for one episode with 1 player and turn on depth map rendering
python basic_track2_supply_gather.py \
--num-episodes 1 \
--record --replay-suffix simple_supply_gather \
--use-depth-map
```

## [basic_track3_supply_battle.py](basic_track3_supply_battle.py)

- Show the basic use of game playing interfaces for track 3
- Two agent is controlled by a simple battle policy that tells the agent to act randomly
- Multiple game configuration parameters (including `timeout`, `map_id`, `walk_speed` etc.) can be set as you wish

```bash
# run the game in Supply_battle mode for one episode with 1 player and turn on depth map rendering
python basic_track3_supply_battle.py \
--num-episodes 1 --num-agents 2 \
--record --replay-suffix simple_supply_battle \
--use-depth-map
```

# Getting Started: baseline training scripts based on Ray

Here we will introduce some applications of using ray to train agents on different tracks.

- A simple PPO reinforcement learning algorithm is used to learn a policy with discrete action spaces.
- You can change the state design or reward function to better adapt to the environment
and train the agent to find the optimal strategy
- Other reinforcement learning algorithms can be used to train the baselines as well.

## [baseline_track1_normal.py](baseline_track1_normal.py)

- Show the basic use of ray to train the agent for track1
- An agent is controlled by a discret policy that tells the agent to walk to target position as possible as quick by trained with a ppo algorithm
- Multiple game configuration parameters (including `timeout`, `map_id`, `walk_speed` etc.) can be set as you wish
- Numbers of environment and training configuration parameters(including `num_worker`,`batch_size`,`max_episode` etc.) can be set as you wish
- You can design yourself environment shape of state, action or reward function

```bash
# run the training iteration for 20 episodes with a single rollout worker 
python baseline_track1_normal.py \
--use-depth-map \
--detailed-log \
--record --replay-suffix baseline_navigation \
--num-workers 1 --stop-iters 20 --stop-reward 99
```

## [baseline_track2_normal.py](baseline_track2_normal.py)

- Show the basic use of ray to train the agent for track2
- An agent is controlled by a learning policy that tells the agent to collect supply as much as possible  by trained with a ppo algorithm
- Multiple game configuration parameters (including `timeout`, `map_id`, `walk_speed` etc.) can be set as you wish
- As well, numbers of environment and training configuration parameters(including `num_worker`,`batch_size`,`max_episode` etc.) can be set as you wish
- You can design yourself environment shape of state, action or reward function

```bash
# run the training iteration for 100 episodes with 10 rollout workers
python baseline_track2_normal.py \
--use-depth-map \
--detailed-log \
--record --replay-suffix baseline_supply \
--num-workers 10 --stop-episodes 100 --train-batch-size 400
```
