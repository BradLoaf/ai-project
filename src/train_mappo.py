import os
import gymnasium as gym
from stable_baselines3.common.callbacks import CheckpointCallback
from multi_agent_wrapper import MultiAgentMetroWrapper
from mappo import MAPPO
from config import num_paths


LOG_DIR = f"logs/MAPPO/"
MODEL_DIR = f"models/MAPPOv3/"
TOTAL_TIMESTEPS = 25_000_000
SAVE_FREQ = 25_000
TB_LOG_NAME = "MAPPO_Metro_Run"
net_arch_config = [256, 128, 128]
policy_kwargs = dict(net_arch=net_arch_config)

# MAPPO configuration
SHARED_POLICY = False
NUM_AGENTS = num_paths


def create_multi_agent_env():
    """Create and wrap the multi-agent environment."""
    env = MultiAgentMetroWrapper(render_mode=None)
    return env


def train_mappo():
    """Initializes and trains the MAPPO agents."""
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    print(f"Initializing MAPPO with {NUM_AGENTS} agents...")
    print(f"Shared policy: {SHARED_POLICY}")
    
    # Create MAPPO model
    model = MAPPO(
        "MlpPolicy",
        multi_agent_env_factory=create_multi_agent_env,
        num_agents=NUM_AGENTS,
        shared_policy=SHARED_POLICY,
        verbose=1,
        tensorboard_log=LOG_DIR,
        device="cpu",
        n_steps=4096,
        learning_rate=1e-5,
        policy_kwargs=policy_kwargs,
        batch_size=64,
        gamma=0.99,
        gae_lambda=0.95,
        n_epochs=10,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
    )
    
    # Create checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=SAVE_FREQ,
        save_path=MODEL_DIR,
        name_prefix="mappo_model",
        save_replay_buffer=True,
    )
    
    print(f"Starting MAPPO training for {TOTAL_TIMESTEPS} timesteps...")
    
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=checkpoint_callback,
        tb_log_name=TB_LOG_NAME
    )
    
    final_model_path = os.path.join(MODEL_DIR, "final_model")
    model.save(final_model_path)
    print(f"Training complete! Final model saved to {final_model_path}")
    
    for agent in model.agents:
        if hasattr(agent, 'env'):
            agent.env.close()


if __name__ == '__main__':
    train_mappo()


