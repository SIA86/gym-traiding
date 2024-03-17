from gymnasium.envs.registration import register

register(
     id="gym_examples/MonoCryptoEnv-v0",
     entry_point="gym_examples.envs:TradingEnv",
     max_episode_steps=1e6,
)