import random
import numpy as np

from typing import List

from scipy.stats import rv_discrete


class VanillaMultiArmedBandit:

    """
    An implementation of the basic multi-armed bandit.
    The reward distributions of the arms must be
    instances of scipy.stats.rv_discrete.
    """

    def __init__(self, n_arms: int) -> None:
        """
        Initializes the number of arms of the bandit to n_arms.
        Creates empty arrays for the total and average rewards,
        and a counter of the
        number of rounds played
        """
        self.n_arms = n_arms
        self.rounds_played = 0
        self.rewards = np.zeros(self.n_arms)
        self.mean_rewards = np.zeros(self.n_arms)

    def __repr__(self) -> str:
        return f"VanillaMultiArmedBandit(n_arms={self.n_arms})"

    def play_round(self, arms: List[rv_discrete]) -> None:
        """
        Simulates one round of the hidden process
        in which the bandit chooses an arm and samples its reward
        """
        chosen_arm_idx = random.choice(range(self.n_arms))
        chosen_arm = arms[chosen_arm_idx]
        self.rewards[chosen_arm_idx] += chosen_arm.rvs()
        self.rounds_played += 1
        self.mean_rewards = self.rewards/self.rounds_played

    def reset_rounds(self) -> None:
        """
        Resets the state of the bandit
        """
        self.rounds_played = 0
        self.rewards = np.zeros(self.n_arms)
        self.mean_rewards = np.zeros(self.n_arms)
