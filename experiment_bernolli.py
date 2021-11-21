import numpy as np

from scipy.stats import bernoulli
from playground.vanilla_mab import VanillaMultiArmedBandit

N_ARMS = 12
N_ROUNDS = N_ARMS*5000

bandit = VanillaMultiArmedBandit(n_arms=N_ARMS)
reward_dists_means = np.random.random(N_ARMS)
arms = [bernoulli(p) for p in reward_dists_means]

for _ in range(N_ROUNDS):
    bandit.play_round(arms)

print(bandit)
print(f"Correlation between the true \
means of the rewards distributions \
and the bandit's mean rewards: \
{np.corrcoef(bandit.mean_rewards, reward_dists_means)[0,1]:.2f}")
