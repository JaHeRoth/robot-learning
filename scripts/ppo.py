# Roll out a batch tau of trajectories following pi for T timesteps
# Walk tau backwards, recording A_t
# Freeze theta_old := theta.copy()
# Compute loss on theta, theta_old, A_t (and maybe s_t for entropy bonus)
