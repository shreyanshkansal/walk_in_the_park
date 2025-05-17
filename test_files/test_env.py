import gym
import dmcgym
import numpy as np
import matplotlib.pyplot as plt

def main():
    env = gym.make("quadruped-run-v0")
    print("Environment loaded: quadruped-run-v0")
    print("Observation space:", env.observation_space)
    print("Action space:", env.action_space)

    obs = env.reset()
    print("Initial observation keys:", obs.keys())
    for key, value in obs.items():
        print(f"{key}: shape = {value.shape}")

    total_reward = 0.0
    max_steps = 100

    plt.ion()  # Interactive mode on
    fig, ax = plt.subplots()

    for step in range(max_steps):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        total_reward += reward

        # Render as image using dm_control physics
        frame = env.unwrapped.physics.render(camera_id=0, height=480, width=640)
        ax.imshow(frame)
        ax.set_title(f"Step {step}")
        plt.pause(0.01)
        ax.clear()

        if done:
            print(f"Episode finished early at step {step}")
            break

    plt.ioff()
    env.close()
    print("Total reward collected:", total_reward)

if __name__ == "__main__":
    main()
