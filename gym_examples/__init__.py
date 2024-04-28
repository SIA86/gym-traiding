from gymnasium.envs.registration import register

register(
     id="Env-v1",
     entry_point="gym_examples.envs:CryptoEnvV1",
     max_episode_steps=1e5,
)

register(
     id="Env-v2",
     entry_point="gym_examples.envs:CryptoEnvV2",
     max_episode_steps=1e5,
)

register(
     id="Env-v3",
     entry_point="gym_examples.envs:CryptoEnvV3",
     max_episode_steps=1e5,
)

register(
     id="Env-v4",
     entry_point="gym_examples.envs:CryptoEnvV4",
     max_episode_steps=1e5,
)
