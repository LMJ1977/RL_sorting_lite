from gym.envs.registration import register

register(
    id='ball-sorting-lite',
    entry_point='ball_sorting_lite.envs:BallSortingEnv_lite',
)
