[![Demo](https://i.imgur.com/xpUow2f.png)](https://youtu.be/W5fCgqlECeI)

# Running an Agent
In the main branch there are 2 options for agents to run. Base PPO and PPO with a GNN feature extractor, these can be found in the models folder. To run an agent run the command `python3 run_agent.py {model_folder}` where `model_folder` is the folder containing the model of your choice. There is also the option to include the deterministic flag before running with `-d` or `--deterministic`. This will cause the model to only select the action with the highest probability rather than sample from a distribution.

# python_mini_plane
This repo uses `pygame` to implement Mini plane, a fun 2D strategic game where you try to optimize the max number of passengers your plane system can handle. Both human and program inputs are supported. One of the purposes of this implementation is to enable reinforcement learning agents to be trained on it.

# Installation
`pip install -r requirements.txt`

# How to run
## To play the game manually
* If you are running for the first time, install the requirements using `pip install -r requirements.txt`
* Activate the virtual environment by running `source myenv/bin/activate`
* Run `python src/main.py`

# Testing
`python -m unittest -v`