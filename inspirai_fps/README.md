# Python Game Play Interface

## Agent Action Variables

- `ActionVariable.WALK_DIR`: the walking direction of the agent
- `ActionVariable.WALK_SPEED`: the walking speed of the agent
- `ActionVariable.TURN_LR_DELTA`: the incremental camera angle of the agent turning left or right
- `ActionVariable.LOOK_UD_DELTA`: the incremental camera angle of the agent looking up or down
- `ActionVariable.JUMP`: the jumping action of the agent
- `ActionVariable.ATTACK`: the shooting action of the agent
- `ActionVariable.RELOAD`: the weapon clip reloading action of the agent
- `ActionVariable.PICKUP`: the action of the agent to pick up a supply

## Agent State Variables

- `StateVariable.LOCATION`: the location of the agent
  - `position_x`: the x coordinate value of the agent's location
  - `position_y`: the y coordinate value of the agent's location (vertical height)
  - `position_z`: the z coordinate value of the agent's location
- `StateVariable.MOVE_DIR`: the moving direction of the agent
  - `move_dir_x`: the x coordinate value of the agent's moving direction
  - `move_dir_y`: the y coordinate value of the agent's moving direction
  - `move_dir_z`: the z coordinate value of the agent's moving direction
- `StateVariable.MOVE_SPEED`: the speed of the agent's movement
- `StateVariable.CAMERA_DIR`: the direction of the camera
  - `pitch`: the vertical angle of the camera
  - `yaw`: the horizontal angle of the camera
- `HEALTH`: the health percentage of the agent
- `WEAPON_AMMO`: the number of bullets left in the agent's weapon clip
- `SPARE_AMMO`: the number of bullets left in the agent's spare ammo
- `IS_ATTACKING`: whether the agent is currently shooting
- `IS_RELOADING`: whether the agent is currently reloading the weapon
- `HIT_ENEMY`: whether the agent hit an enemy
- `HIT_BY_ENEMY`: whether the agent is hit by an enemy
- `NUM_SUPPLIES`: the number of supplies collected by the agent
- `TARGET_LOCATION`: the location of the agent's target
  - `position_x`: the x coordinate value of the target location
  - `position_y`: the y coordinate value of the target location (vertical height)
  - `position_z`: the z coordinate value of the target location
- `IS_WAITING_RESPAWN`: whether the agent is waiting for respawn
- `IS_INVINCIBLE`: whether the agent is invincible

## Game Modes

- `Game.MODE_NAVIGATION`: the track 1 mode identifier
- `Game.MODE_SUP_GATHER`: the track 2 mode identifier
- `Game.MODE_SUP_BATTLE`: the track 3 mode identifier

## Game Configuration

Users can change the game configuration by using the following methods and the new game configuration will be applied to the next game when calling `Game.new_episode()`.

- `Game.get_game_config`: get all game configuration parameters as a dictionary
- `Game.set_episode_timeout`: set the episode timeout in seconds
- `Game.set_map_id`: set the map id
- `Game.set_game_mode`: set the game mode
- `Game.set_random_seed`: set the random seed (used by supply generation and agent spawning)
- `Game.set_start_location`: set the start location of the specified agent
- `Game.set_target_location`: set the target location of the Navigation mode
- `Game.set_available_actions`: set the available action variables of the agent
- `Game.set_game_replay_suffix`: set the suffix of the game replay filename
- `Game.set_supply_heatmap_center`: set the center of the initial heatmap of supply distribution
- `Game.set_supply_heatmap_radius`: set the radius of the initial heatmap of supply distribution
- `Game.set_supply_outdoor_richness`: control the abundance of supply in the open field in percentage
- `Game.set_supply_indoor_richness`: control the abundance of supply inside houses in percentage
- `Game.set_supply_spacing`: control the spacing between supply in the open field in meters
- `Game.set_supply_outdoor_quantity_range`: control the quantity range of a supply in the open field
- `Game.set_supply_indoor_quantity_range`: control the quantity range of a supply inside houses
- `Game.add_supply_refresh`: general interface to add a supply refresh event in the game
  - `refresh_time`: the time of the refresh event
  - `heatmap_center`: the center of the heatmap of supply distribution
  - `heatmap_radius`: the radius of the heatmap of supply distribution
  - `outdoor_richness`: control the abundance of supply in the open field in percentage
  - `indoor_richness`: control the abundance of supply inside houses in percentage
- `Game.add_agent`: general interface to add an agent in the game
  - `agent_name`: the name of the added agent
  - `health`: the health point (HP) of the agent
  - `start_location`: the agent's initial spawning location
  - `num_clip_ammo`: the number of bullets in the agent's weapon clip
  - `num_pack_ammo`: the number of bullets in the agent's spare ammo
  - `attack`: the attack power of the agent
- `Game.turn_on_record`: turn on recording of the game replay
- `Game.turn_off_record`: turn off recording of the game replay
- `Game.turn_on_depth_map`: turn on computing of the depth map in the agent state
- `Game.turn_off_depth_map`: turn off computing of the depth map of the agent state

## Game Workflow

- `Game.init`: initialize the game server and pull up the backend game engine to connect to the game server
- `Game.new_episode`: start a new episode of the game with all agent and game states reset to the initial ones
- `Game.make_action`: send actions of agents (in `dict`) from the game server to the backend game engine
- `Game.get_state`: get the agent state (by id) from the backend game engine
- `Game.get_state_all`: get all agent states (as `dict`) from the backend game engine
- `Game.is_episode_finished`: check whether the current running episode is finished
- `Game.close`: close the game server and shutdown the backend game engine
