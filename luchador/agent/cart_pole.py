"""Implementation of CartPole Agent from Sutton

https://webdocs.cs.ualberta.ca/~sutton/book/code/pole.c
"""
from __future__ import division
from __future__ import absolute_import

import numpy as np

from . base import BaseAgent

_1_DEG = 0.0174532
_6_DEG = 6 * _1_DEG
_12_DEG = 12 * _1_DEG
_15_DEG = 15 * _1_DEG

N_BOX = 162


def _get_box(x, x_dot, theta, theta_dot):
    if abs(x) > 2.4 or abs(theta) > _12_DEG:
        return -1

    box = 0
    if x < -0.8:
        pass
    elif x < 0.8:
        box = 1
    else:
        box = 2

    if x_dot < -0.5:
        pass
    elif x_dot < 0.5:
        box += 3
    else:
        box += 6

    if theta < -_6_DEG:
        pass
    elif theta < -_1_DEG:
        box += 9
    elif theta < 0:
        box += 18
    elif theta < _1_DEG:
        box += 27
    elif theta < _6_DEG:
        box += 36
    else:
        box += 45

    if theta_dot < -_15_DEG:
        pass
    elif theta_dot < _15_DEG:
        box += 54
    else:
        box += 108

    return box


def _truncated_sigmoid(s):
    return 1. / (1. + np.exp(-max(-50., min(s, 50.))))


class CartPoleAgent(BaseAgent):
    """Implementation of CartPole Agent from Sutton

    https://webdocs.cs.ualberta.ca/~sutton/book/code/pole.c
    """
    def __init__(self,
                 action_lr=1000,
                 critic_lr=0.5,
                 critic_discount=0.95,
                 action_decay=0.9,
                 critic_decay=0.8):

        self.action_lr = action_lr
        self.critic_lr = critic_lr
        self.critic_discount = critic_discount
        self.action_decay = action_decay
        self.critic_decay = critic_decay

        self.action_weight = np.zeros((N_BOX,))
        self.critic_weight = np.zeros((N_BOX,))
        self.action_eligibility = np.zeros((N_BOX,))
        self.critic_eligibility = np.zeros((N_BOX,))

        self.box = 0

    def init(self, env):
        pass

    def reset(self, observation):
        self.box = _get_box(**observation)
        self.action_eligibility = np.zeros((N_BOX,))
        self.critic_eligibility = np.zeros((N_BOX,))

    def learn(self, state0, action, reward, state1, terminal, info=None):
        update = action - 0.5
        self.action_eligibility[self.box] += (1.0 - self.action_decay) * update
        self.critic_eligibility[self.box] += (1.0 - self.critic_decay)

        p_prev = self.critic_weight[self.box]
        self.box = _get_box(**state1)
        p_current = 0.0 if terminal else self.critic_weight[self.box]

        r_hat = reward + self.critic_discount * p_current - p_prev

        self.action_weight += self.action_lr * r_hat * self.action_eligibility
        self.critic_weight += self.critic_lr * r_hat * self.critic_eligibility

        self.action_eligibility *= self.action_decay
        self.critic_eligibility *= self.critic_decay

    def act(self):
        prob = _truncated_sigmoid(self.action_weight[self.box])
        return int(np.random.rand() < prob)

    def perform_post_episode_task(self, stats):
        pass
