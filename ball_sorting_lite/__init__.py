from gym.envs.registration import register

register(
    id='ball-lite',
    entry_point='ball_sorting_lite.envs:BallSortingEnv_lite',
)
