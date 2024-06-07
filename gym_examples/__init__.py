from gymnasium.envs.registration import register

register(
     id="EnvTrain-v1",
     entry_point="gym_examples.envs:EnvTrain",
     max_episode_steps=1e6,
)

register(
     id="EnvVal-v1",
     entry_point="gym_examples.envs:EnvVal",
     max_episode_steps=1e6,
)
