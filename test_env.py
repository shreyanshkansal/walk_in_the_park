import gym
import dmcgym
import numpy as np

def main():
    # Create the environment
    env = gym.make("quadruped-run-v0")

    print("Environment loaded: quadruped-run-v0")
    print("Observation space:", env.observation_space)
    print("Action space:", env.action_space)

    obs = env.reset()
    print("Initial observation keys:", obs.keys())
    for key, value in obs.items():
        print(f"{key}: shape = {value.shape}")

    total_reward = 0.0
    max_steps = 500

    for step in range(max_steps):
        action = env.action_space.sample()
        if step < 5:
            print(f"Sample action at step {step}: {action}")

        obs, reward, done, info = env.step(action)
        total_reward += reward

        # env.render()  # Uncomment to see simulation (if your system supports it)

        if done:
            print(f"Episode finished early at step {step}")
            break

    print("Total reward collected:", total_reward)
    env.close()

if __name__ == "__main__":
    main()
