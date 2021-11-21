import random

from typing import List

from scipy.stats import rv_discrete

from playground.vanilla_mab import VanillaMultiArmedBandit


class EpsilonGreedyMAB(VanillaMultiArmedBandit):
    """
    A multi-armed bandit using an epsilon-greedy
    strategy for choosing an optimal arm
    """

    def __init__(self, n_arms: int, epsilon=.05) -> None:
        super().__init__(n_arms)
        self.epsilon = epsilon

    def __repr__(self) -> str:
        return f"EpsilonGreedyMAB(n_arms={self.n_arms})"

    def play_round(self, arms: List[rv_discrete]) -> None:
        """
        Simulates a round using an epsilon-greedy
        approach
        """
        prob = random.random()  # sample from [0, 1)

        if prob <= self.epsilon:  # explore
            chosen_arm_idx = random.choice(range(self.n_arms))
        else:  # exploit
            chosen_arm_idx = self.mean_rewards.argmax()

        chosen_arm = arms[chosen_arm_idx]
        self.rewards[chosen_arm_idx] += chosen_arm.rvs()
        self.rounds_played += 1
        self.mean_rewards = self.rewards/self.rounds_played
