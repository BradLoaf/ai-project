import os
import argparse
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from pilot_planning_env import PlaneGameEnv

def run_agent(model_folder):
    model_path = os.path.join(model_folder, "final_model.zip")
    normalize_path = os.path.join(model_folder, "vec_normalize.pkl")

    if not os.path.exists(model_path):
        print(f"Error: No model files found in {model_folder}")
        return
        
    if not os.path.exists(normalize_path):
        print(f"Error: 'vec_normalize.pkl' not found at {normalize_path}")
        return

    def create_eval_env():
        env = PlaneGameEnv(render_mode="human")
        env = gym.wrappers.TimeLimit(env, max_episode_steps=5000)
        return env

    env = DummyVecEnv([create_eval_env])

    env = VecNormalize.load(normalize_path, env)
    env.training = False
    env.norm_reward = False

    model = PPO.load(model_path, env=env)
    obs = env.reset()
    total_reward = 0
    
    while True:
        action, states = model.predict(obs, deterministic=False)
        obs, reward, terminated, info = env.step(action)
        
        total_reward += reward[0]
        if terminated[0]:
            print(f"Total Reward for episode: {total_reward}")
            total_reward = 0
            obs = env.reset()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a trained Mini Plane PPO agent with UI.")
    parser.add_argument("model_folder", help="directory containing the saved model (final_model.zip) and normalizer (vec_normalize.pkl).")
    
    args = parser.parse_args()
    
    run_agent(args.model_folder)