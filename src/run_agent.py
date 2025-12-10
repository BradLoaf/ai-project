import os
import argparse
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from collections import Counter
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
    invalid_actions = 0
    passengers_delivered = 0
    actions_taken = Counter()
    
    while True:
        action, states = model.predict(obs, deterministic=False)
        obs, reward, terminated, info_list = env.step(action)
        
        info = info_list[0]
        
        total_reward += reward[0]
        
        if not info.get("valid_action", True):
            invalid_actions += 1
            
        passengers_delivered += info.get("passengers_delivered", 0)
        
        action_type = info.get("action_type", "UNKNOWN")
        actions_taken[action_type] += 1

        if terminated[0]:
            print("-" * 30)
            print(f"Episode Finished")
            print(f"Total Reward:       {total_reward:.2f}")
            print(f"Total Passengers:   {passengers_delivered}")
            print(f"Invalid Moves:      {invalid_actions}")
            print(f"Action Distribution:")
            for action, count in actions_taken.items():
                print(f"  - {action}: {count}")
            print("-" * 30)
            
            total_reward = 0
            invalid_actions = 0
            passengers_delivered = 0
            actions_taken = Counter()
            obs = env.reset()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a trained Mini Plane PPO agent with UI.")
    parser.add_argument("model_folder", help="directory containing the saved model (final_model.zip) and normalizer (vec_normalize.pkl).")
    
    args = parser.parse_args()
    
    run_agent(args.model_folder)
