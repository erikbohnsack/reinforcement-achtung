from gym.envs.registration import register

register(
    id='AchtungDieKurve-v1',
    entry_point='gym_achtung.envs:AchtungDieKurve',
    timestep_limit=100000,
    reward_threshold=1.0,
    nondeterministic=True,
)

register(
    id='AchtungDieKurveRandomOpponent-v1',
    entry_point='gym_achtung.envs:AchtungDieKurveRandomOpponent',
    timestep_limit=100000,
    reward_threshold=1.0,
    nondeterministic=True,
)

register(
    id='AchtungDieKurveFullImage-v1',
    entry_point='gym_achtung.envs:AchtungDieKurveFullImage',
    timestep_limit=100000,
    reward_threshold=1.0,
    nondeterministic=True,
)