import numpy as np
from typing import Callable


Policy = Callable[[np.ndarray, np.ndarray, int], int]
Plot = tuple[np.ndarray, dict[str, str|float]]


def epsilon_greedy(estimates: np.ndarray[float], eps: float) -> int:
    """
    simulates the epsilon-greedy method of choosing an arm
    :param estimates: the current estimates of the arms
    :param eps: epsilon - the probability of choosing a random arm
    :return: the index of the chosen arm
    """

    if np.random.random() < eps:
        return np.random.randint(estimates.shape[0])
    else:
        return np.argmax(estimates)


def UCB(estimates: np.ndarray[float], counts: np.ndarray[int], t: int, c: float) -> int:
    """
    simulates the upper confidence bound method of choosing an arm
    :param estimates: the current estimates of the arms
    :param counts: the number of times each arm has been chosen
    :param c: the confidence level
    :param t: the current time step
    :return: the index of the chosen arm
    """

    return np.argmax(estimates + c * np.sqrt(np.log(t) / counts))


def gradient_bandit(k: int, H: np.ndarray[float]) -> int:
    """
    simulates the gradient bandit method of choosing an arm
    :param k: the number of arms
    :param H: the current preference parameter of each arm
    :return: the index of the chosen arm
    """

    return np.random.choice(k, p=(np.exp(H) / np.sum(np.exp(H))))


def epsiode(T: int, arms: np.ndarray, policy: Policy) -> tuple[np.ndarray]:
    """
    simulates an episode of T time steps
    :param T: the number of time steps
    :param arms: the arms to choose from
    :param policy:
        the policy to use for choosing arms
        :param estimates: the current estimates of the arms
        :param count: the number of times each arm has been chosen
        :param t: the current time step
        :return: the index of the chosen arm
    :return: the rewards received at each time step
    :return: the actions chosen at each time step
    :return: the percentage of times the optimal action was chosen at each time step
    """

    estimates = np.zeros(arms.shape[0])
    rewards = np.zeros(T)
    actions = np.zeros(T)
    optimal_actions = np.zeros(T)
    counts = np.zeros(arms.shape[0])

    greedy = np.argmax(arms[:, 0])

    for t in range(T):
        arm = policy(estimates, t=t, counts=counts)
        reward = np.random.normal(arms[arm, 0], arms[arm, 1])

        rewards[t] = reward
        actions[t] = arm
        optimal_actions[t] = actions[t] == greedy

        counts[arm] += 1
        estimates[arm] = estimates[arm] + (1/counts[arm]) * (reward - estimates[arm])

    return rewards, actions, optimal_actions


def gradient_episode(T: int, arms: np.ndarray, alpha: float, baseline: bool) -> tuple[np.ndarray]:
    """
    simulates an episode of stochastic gradient ascent on parametric policy space for T time steps
    :param T: the number of time steps
    :param arms: the arms to choose from
    :param alpha: the step size
    :param baseline: whether to use a baseline
    :return: the rewards received at each time step
    :return: the actions chosen at each time step
    :return: the percentage of times the optimal action was chosen at each time step
    """

    rewards = np.zeros(T)
    actions = np.zeros(T)
    optimal_actions = np.zeros(T)
    H = np.zeros(arms.shape[0])
    Rt = 0.0

    greedy = np.argmax(arms[:, 0])

    for t in range(T):
        arm = gradient_bandit(arms.shape[0], H)
        reward = np.random.normal(arms[arm, 0], arms[arm, 1])
        Rt = Rt + reward

        rewards[t] = reward
        actions[t] = arm
        optimal_actions[t] = actions[t] == greedy

        pi = -np.exp(H) / np.sum(np.exp(H))
        pi[arm] = pi[arm] + 1
        H = H + alpha * (reward - baseline*Rt/(t+1)) * pi

    return rewards, actions, optimal_actions


def parallel_episodes(N: int, T: int, k: int, policies: list[Policy], std: float) -> tuple[np.ndarray]:
    """
    simulates N episodes of T time steps in parallel
    :param N: the number of episodes
    :param T: the number of time steps
    :param k: the number of arms
    :param policies: a list of policies to use for choosing arms
    :param std: the standard deviation of the arms
    :return average_reward: the average reward received at each time step for each policy
    :return optimal_action: the percentage of times the optimal action was chosen at each time step for each policy
    """

    average_reward = np.zeros((T, len(policies)))
    optimal_action = np.zeros((T, len(policies)))

    for _ in range(N):
        arms = np.stack((np.random.normal(0, 1, k), np.full(k, std)), axis=1)
        rewards = np.zeros((T, len(policies)))
        optimal = np.zeros((T, len(policies)))

        for i, policy in enumerate(policies):
            rewards[:, i], _, optimal[:, i] = epsiode(T, arms, policy)

        average_reward += rewards
        optimal_action += optimal

    return average_reward/N, optimal_action*100/N


def parallel_gradient_episodes(N: int, T: int, k: int, alpha: list[float], baseline: list[bool]) -> tuple[np.ndarray]:
    """
    simulates N episodes of stochastic gradient ascent on parametric policy space for T time steps in parallel
    :param N: the number of episodes
    :param T: the number of time steps
    :param k: the number of arms
    :param alpha: a list of the step sizes
    :param baseline: a list of whether to use a baseline in each run
    :return average_reward: the average reward received at each time step for each policy
    :return optimal_action: the percentage of times the optimal action was chosen at each time step for each policy
    """

    average_reward = np.zeros((T, len(alpha), len(baseline)))
    optimal_action = np.zeros((T, len(alpha), len(baseline)))

    for _ in range(N):
        arms = np.stack((np.random.normal(4, 1, k), np.full(k, 1)), axis=1)
        rewards = np.zeros((T, len(alpha), len(baseline)))
        optimal = np.zeros((T, len(alpha), len(baseline)))

        for i, baseline_i in enumerate(baseline):
            for j, alpha_j in enumerate(alpha):
                rewards[:, i, j], _, optimal[:, i, j] = gradient_episode(T, arms, alpha_j, baseline_i)

        average_reward += rewards
        optimal_action += optimal

    return average_reward/N, optimal_action*100/N