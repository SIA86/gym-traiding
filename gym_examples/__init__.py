from gymnasium.envs.registration import register

register(
     id="CryptoEnvQuantile-v1",
     entry_point="gym_examples.envs:CryptoEnvQuantile_v1",
     max_episode_steps=1e6,
)

register(
     id="CryptoEnvMinMaxScaler-v1",
     entry_point="gym_examples.envs:CryptoEnvMinMaxScaler_v1",
     max_episode_steps=1e6,
)

register(
     id="CryptoEnvQuantile-v2",
     entry_point="gym_examples.envs:CryptoEnvQuantile_v2",
     max_episode_steps=1e6,
)

register(
     id="CryptoEnvMinMaxScaler-v2",
     entry_point="gym_examples.envs:CryptoEnvMinMaxScaler_v2",
     max_episode_steps=1e6,
)
