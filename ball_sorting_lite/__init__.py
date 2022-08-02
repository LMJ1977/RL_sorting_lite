from gym.envs.registration import register

register(
    id='ball-sorting-lite-v0',
    entry_point='ball_sorting_lite.envs:BallSortingLiteEnv',
)
