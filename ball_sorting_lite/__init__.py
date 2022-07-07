from gym.envs.registration import register

register(
    id='ball_sorting-v0_lite',
    entry_point='ball_sorting_lite.envs:BallSortingEnv_lite',
)
