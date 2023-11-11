import numpy as np
from typing import Callable


def episode(
        states: np.ndarray, actions: np.ndarray, pi: Callable[[int], int],
        rewards: Callable[[int, int], float]
    ) -> tuple[np.ndarray]:
    """
    Generate an episode starting from a random state using the given policy.
    Returns the sequence of states, actions, and rewards.
    """
    S0 = np.random.choice(states[1:-1])
    S, A, R = [S0], [], [np.nan]
    while S[-1] not in (0, 6):
        A.append(a := pi(S[-1]))
        R.append(rewards(S[-1], a))
        S.append(S[-1] + a)
    return np.array(S[:-1]), np.array(A), np.array(R)


def TD0(
        states: np.ndarray, rewards: Callable[[int, int], float],
        pi: Callable[[int], int], gamma: float, alpha: float, V: np.ndarray|None = None
    ) -> np.ndarray:
    """
    Generate an episode and use it to update the value function using TD0
    """

    V = np.zeros(len(states)) if V is None else V
    S = np.random.choice(states[1:-1])
    while S not in (0, 6):
        R = rewards(S, (A := pi(S)))
        V[S] += alpha * (R + gamma*V[S_ := S + A] - V[S])
        S = S_
    return V


def MC(
        states: np.ndarray, actions: np.ndarray, rewards: Callable[[int, int], float],
        pi: Callable[[int], int], gamma: float, alpha: float, V: np.ndarray|None = None
    ) -> np.ndarray:
    """
    Generate an episode and use it to update the value function using MC
    """

    V = np.zeros(len(states)) if V is None else V
    S, A, R = episode(states, actions, pi, rewards)
    G = 0
    for t in range(len(S)-1, -1, -1):
        G = gamma*G + R[t+1]
        V[S[t]] += alpha * (G - V[S[t]])
    return V


def RMS_error(estimate: np.ndarray, true: np.ndarray) -> float:
    """Return RMS error between estimate and true values."""
    return np.sqrt(np.mean((estimate - true)**2))