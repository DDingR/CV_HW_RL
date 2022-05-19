import gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from mlagents_envs.environment import UnityEnvironment
from gym_unity.envs import UnityToGymWrapper

def main():
    env_path = './envs/3dBall_headless/3dBall_headless.x86_64'

    unity_env = UnityEnvironment(env_path)
    env = UnityToGymWrapper(
        unity_env
        )
    # logger.configure('./logs')

    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(
        total_timesteps=2500000
        )
    print("saving model")
    model.save("ppo_baselines3_test.pkl")

if __name__ == '__main__':
    main()