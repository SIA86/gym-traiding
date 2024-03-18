from gymnasium.envs.registration import register

register(
     id="CryptoMono-v0",
     entry_point="gym_examples.envs:TradingEnv",
     max_episode_steps=1e5,
)

register(
     id="CryptoMono-v1",
     entry_point="gym_examples.envs:TradingEnv1",
     max_episode_steps=1e5,
)