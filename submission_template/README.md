# Submitting solutions for online evaluation

Here we provide a template for you to easily pack your solution and upload it to our online evaluation system.

## Template Structure

```bash
submission_template
├── Dockerfile               # Dockerfile for building the submission image
├── common.py                # Common functions and variables
├── eval.py                  # Main evaluation script
├── eval_track_1_1.py        # Evaluation function for track 1.1
├── eval_track_1_2.py        # Evaluation function for track 1.2
├── eval_track_2.py          # Evaluation function for track 2
├── requirements.txt         # Additional python packages required by the submission 
└── submission               # Submission source code and data
    ├── __init__.py          # Making the submission folder a python package
    ├── other files or data  # Other files or data
    └── agents.py            # Agent classes for the 3 tracks
```

## Implement your solution

- Modify the code in `submission/agents.py` to implement your agents. Below is an example of how to implement a simple navigation agent.

    ```python
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
    ```

- You can also add additional files (`.py` modules or data) to the `submission` folder and import them in `submission/agents.py`
- If you need additional python packages, add them to the `requirements.txt` file

    ```bash
    pip freeze > requirements.txt
    ```

- If you have other non-python dependencies, add installation commands in the `Dockerfile` file

## Test your solution locally

- Once your finished up dealing with the submission folder, you can test your solution locally by running the following command (for example):

    ```bash
    # make sure your are in the root of this template folder
    python eval.py --track 1a \
    --local-test --map-dir /path/to/map-data --engine-dir /path/to/backend-engine \
    --map-list 1 2 3 --episodes-per-map 2 --episode-timeout 10
    ```

## Submit your solution

Once you are satisfied with your solution, you can submit your solution following the steps below:

- pack your solution into a zip file

    ```bash
    # make sure your are in the root of this template folder
    zip -r /path/to/submission.zip * 
    ```

- upload your solution to our [online evaluation system](https://wildscav-eval.inspirai.com) (the maximum size of the zip file is limited 500MB)

## Important Notes

- Do not modify the `eval.py` file.
- Modify the `Dockerfile` file only when you need to add additional non-python dependencies.
- Evaluation scripts for 3 tracks (`eval_track_*.py`) are only for your reference. You can use them to test your solution locally. The actual evaluation code may be different.
