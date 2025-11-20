import os
import platform
import gymnasium as gym
import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from mini_metro_env import MetroGameEnv
from gnn_extractor import GNNFeatureExtractor

LOG_DIR = f"logs/GNN-10/"
MODEL_DIR = f"models/PPO/GNN-10/"
TOTAL_TIMESTEPS = 25_000_000
SAVE_FREQ = 50_000
TB_LOG_NAME = "PPO_GATv2_BN"

def linear_schedule(initial_value: float):
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func

def mask_fn(env: gym.Env) -> np.ndarray:
    """Bridge function for ActionMasker."""
    return env._get_action_mask()

def create_env():
    """Creates the env and wraps it with ActionMasker."""
    env = MetroGameEnv(render_mode=None)
    env = ActionMasker(env, mask_fn) 
    return env

def train_agent():
    if __name__ == '__main__':
        os.makedirs(LOG_DIR, exist_ok=True)
        os.makedirs(MODEL_DIR, exist_ok=True)
        
        num_cpu = os.cpu_count() - 1 if os.cpu_count() > 1 else 1
        start_method = 'fork' if platform.system() != 'Windows' else 'spawn'
        
        vec_env = make_vec_env(
            create_env,
            n_envs=num_cpu,
            vec_env_cls=SubprocVecEnv,
            vec_env_kwargs=dict(start_method=start_method)
        )

        vec_env = VecNormalize(
            vec_env, 
            norm_obs=False, 
            norm_reward=True, 
            clip_reward=10.
        )

        checkpoint_callback = CheckpointCallback(
            save_freq=SAVE_FREQ,
            save_path=MODEL_DIR,
            name_prefix="metro_gnn_bn",
            save_replay_buffer=True,
            save_vecnormalize=True,
        )

        policy_kwargs = dict(
            features_extractor_class=GNNFeatureExtractor,
            features_extractor_kwargs=dict(features_dim=128),
        )

        model = MaskablePPO(
            "MultiInputPolicy",
            vec_env,
            verbose=1,
            tensorboard_log=LOG_DIR,
            device="auto",
            n_steps=6096,
            policy_kwargs=policy_kwargs,
            learning_rate=linear_schedule(3e-4),
            batch_size=3096,
            gamma=0.9999,
            gae_lambda=0.98,
            n_epochs=10,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.3,
            clip_range=0.2
        )

        print(f"Starting GNN (GATv2 + BatchNorm) training on {num_cpu} cores...")

        try:
            model.learn(
                total_timesteps=TOTAL_TIMESTEPS,
                callback=checkpoint_callback,
                tb_log_name=TB_LOG_NAME
            )
        except KeyboardInterrupt:
            print("\nTraining interrupted by user. Saving current progress...")
        except Exception as e:
            print(f"\nAn error occurred: {e}")
            raise e
        finally:
            final_model_path = os.path.join(MODEL_DIR, "final_model")
            model.save(final_model_path)
            vec_env.save(os.path.join(MODEL_DIR, "vec_normalize.pkl"))
            print(f"Training ended. Final model saved to {final_model_path}")
            vec_env.close()

if __name__ == '__main__':
    train_agent()