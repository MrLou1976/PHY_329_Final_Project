
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from TokamakModel import TokamakEnv
        
def main():
    target_current = float(input('Enter the plasma current you want to achieve: '))

    # Create the environment
    env = make_vec_env(lambda: TokamakEnv(target_current=target_current), n_envs=4)

    # Define the RL model (tuned for faster execution)
    model = PPO("MlpPolicy", env, policy_kwargs={'net_arch': [32]}, verbose=1, n_epochs=1, n_steps=2)

    # Training
    model.learn(total_timesteps=1000)  

    model.save("tokamak_control_policy")

    ############### TEST THE MODEL ####################

    env = TokamakEnv(target_current=target_current)
    obs = env.reset()
    done = False

    # Run the model to control the tokamak system
    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(action)
        print(f"Obs: {obs}, Reward: {reward}")

    # Extract the final observed values
    final_current = obs[0]


    # Calculate errors
    current_error = abs(final_current - target_current)


    print(f"Final current: {final_current} A")

    print(f"Target current: {target_current} A")


    print(f"Current error: {current_error}")


    # Check if the observed results are within acceptable tolerance
    current_tolerance = 1e-4  

    if current_error <= current_tolerance:
        print("Current is within the acceptable tolerance.")
    else:
        print("Current is out of tolerance.")

if __name__ == "main":
    main()

