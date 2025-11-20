import os
import gymnasium as gym
import torch
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# Import your custom classes (Critical for pickle loading)
from mini_metro_env import MetroGameEnv
from gnn_extractor import GNNFeatureExtractor 

# --- Configuration ---
MODEL_DIR = "models/PPO/GNN-10/" 
MODEL_PATH = os.path.join(MODEL_DIR, "final_model.zip")
STATS_PATH = os.path.join(MODEL_DIR, "vec_normalize.pkl")

def mask_fn(env: gym.Env):
    return env._get_action_mask()

def main():
    print("Initializing Environment...")

    def make_env():
        # Ensure human rendering is enabled
        env = MetroGameEnv(render_mode="human")
        env = ActionMasker(env, mask_fn)
        return env

    # DummyVecEnv is required because the model was trained on a VecEnv
    env = DummyVecEnv([make_env])

    # 1. Load Normalization Statistics
    if os.path.exists(STATS_PATH):
        print(f"Loading normalization stats from {STATS_PATH}...")
        env = VecNormalize.load(STATS_PATH, env)
        env.training = False     # Do not update stats during inference
        env.norm_reward = False  # See raw score in logs
    else:
        print("Warning: No normalization stats found. Agent might perform poorly.")

    # 2. Load the Trained Agent
    print(f"Loading model from {MODEL_PATH}...")
    
    # Force load to CPU for inference (usually smoother for rendering)
    # Change to "cuda" if you want, but unnecessary for single-env inference
    custom_objects = {
        "learning_rate": 0.0,
        "lr_schedule": lambda _: 0.0,
        "clip_range": lambda _: 0.0,
    }
    
    try:
        model = MaskablePPO.load(MODEL_PATH, custom_objects=custom_objects, device="cpu")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Did you ensure 'gnn_extractor.py' matches the training code exactly?")
        return

    print("\n--- Starting Simulation (Press Ctrl+C to stop) ---")
    
    obs = env.reset()
    
    try:
        while True:
            action_masks = env.env_method("action_masks")[0]
            
            # deterministic=True picks the absolute best move (High Score Mode)
            # deterministic=False picks probabalistically (Creative Mode)
            action, _states = model.predict(
                obs, 
                action_masks=action_masks, 
                deterministic=True 
            )

            obs, rewards, dones, infos = env.step(action)

            # Render happens inside step() automatically for "human" mode in your Env class
            
            if dones[0]:
                info = infos[0]
                print(f"Game Over! Final Score: {info.get('score', 'Unknown')}")
                # Obs is automatically reset by VecEnv, game continues immediately

    except KeyboardInterrupt:
        print("Simulation stopped by user.")
    finally:
        env.close()

if __name__ == "__main__":
    main()
