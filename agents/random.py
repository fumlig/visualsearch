class RandomAgent():
    def __init__(self, env):
        self.action_space = env.action_space

    def predict(self, _obs):
        return self.action_space.sample()
