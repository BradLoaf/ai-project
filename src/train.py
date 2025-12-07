import os
import gymnasium as gym
import platform
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.vec_env import VecNormalize
from GNN_feature_extractor import FeatureExtractor
from pilot_planning_env import PlaneGameEnv

LOG_DIR = f"logs/GNN_plane_run/"
MODEL_DIR = f"models/PPO/GNN_plane_run/"
TOTAL_TIMESTEPS = 25_000_000
SAVE_FREQ = 25_000

TB_LOG_NAME = "GNN_plane_run"

policy_kwargs = dict(
    features_extractor_class=FeatureExtractor,
    features_extractor_kwargs=dict(features_dim=128),
    net_arch=dict(pi=[128, 128], vf=[128, 128]) 
)

def create_env():
    """
    Helper function to create and wrap the environment
    Includes a timeout incase the mediator breaks
    """
    env = PlaneGameEnv(render_mode=None)
    env = gym.wrappers.TimeLimit(env, max_episode_steps=5000)
    return env

def train_agent():
    """
    Initializes and trains the PPO agent
    Creates seperate enviornments for each core on the CPU to speed up training
    """
    if __name__ == '__main__':
        os.makedirs(LOG_DIR, exist_ok=True)
        os.makedirs(MODEL_DIR, exist_ok=True)
        # Gemini helped with the make_vec_env and the start_method for parallel training
        num_cpu = os.cpu_count() if os.cpu_count() > 1 else 1
        start_method = 'fork' if platform.system() != 'Windows' else 'spawn'
        env = make_vec_env(
            create_env,
            n_envs=num_cpu,
            vec_env_cls=SubprocVecEnv,
            vec_env_kwargs=dict(start_method=start_method)
        )

        # This turns all inputs and rewards into a Z-score
        # This is very useful in PPO to prevent the network from being overwhelmed
        env = VecNormalize(env, gamma=0.999)

        checkpoint_callback = CheckpointCallback(
            save_freq=SAVE_FREQ,
            save_path=MODEL_DIR,
            name_prefix="plane_rl_model",
            save_replay_buffer=True,
            save_vecnormalize=True,
        )

        model = PPO(
            "MultiInputPolicy",
            env,
            verbose=1,
            tensorboard_log=LOG_DIR,
            device="cpu",
            n_steps=2048,
            learning_rate=3e-4,
            policy_kwargs=policy_kwargs,
            batch_size=4096,
            gamma=0.999,
            gae_lambda=0.95,
            n_epochs=10,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
        )

        print(f"Starting training on {num_cpu} cores for {TOTAL_TIMESTEPS} timesteps...")

        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=checkpoint_callback,
            tb_log_name=TB_LOG_NAME
        )

        final_model_path = os.path.join(MODEL_DIR, "final_model")
        model.save(final_model_path)
        env.save(os.path.join(MODEL_DIR, "vec_normalize.pkl"))
        print(f"Training complete! Final model saved to {final_model_path}")

        env.close()

if __name__ == '__main__':
    train_agent()
