import gym
import numpy as np
import cv2 as cv


class SignalEnv(gym.Env):

    metadata = {"render.modes": ["ansi"]}

    def __init__(self, signal, window, delta=None):
        self.signal = signal
        self.shape = signal.shape
        self.ndim = len(self.shape)
        self.window = window
        self.delta = delta if delta is not None else np.ones(self.ndim, dtype=int)

        assert self.ndim == len(self.window)
        assert self.ndim == len(self.delta)
        assert all([window_dim <= shape_dim for shape_dim, window_dim in zip(self.shape, self.window)])

        self.reward_range = (-np.inf, np.inf)
        self.action_space = gym.spaces.Discrete(self.ndim*2)
        self.observation_space = gym.spaces.Box(0, 1, self.window)

    def reset(self):
        self.position = np.zeros(self.ndim, dtype=int)
        return self.observation()

    def step(self, action):
        dim = action // 2
        neg = action % 2 == 0

        assert dim < self.ndim

        if neg and self.position[dim] - self.delta[dim] >= 0:
            self.position[dim] -= self.delta[dim]
        elif not neg and self.position[dim] + self.delta[dim] <= self.shape[dim] - self.window[dim]:
            self.position[dim] += self.delta[dim]

        return self.observation(), 0.0, False, {}

    def render(self, mode="ansi"):
        print("shape:", self.shape)
        print("window:", self.window)
        print("position:", self.position)

    def close(self):
        pass

    def seed(self, seed=None):
        self.random = np.random.default_rng(seed)
        return [seed]

    def observation(self):
        visible = tuple(slice(self.position[dim], self.position[dim] + self.window[dim]) for dim in range(len(self.shape)))
        return self.signal[visible]


if __name__ == "__main__":
    img = cv.imread("data/lenna.png")

    env = SignalEnv(img, (256, 256, 1), delta=(32, 32, 2))
    obs = env.reset()
    done = False

    cv.imshow("signal", obs)
    cv.waitKey(1)
    env.render()

    while not done:
        action = int(input())
        obs, reward, done, _info = env.step(action)

        cv.imshow("signal", obs)
        cv.waitKey(1)
        env.render()