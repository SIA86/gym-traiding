from gymnasium.envs.registration import register

register(
     id="CryptoEnv-v1",
     entry_point="gym_examples.envs:CryptoEnv",
     max_episode_steps=1e6,
)

