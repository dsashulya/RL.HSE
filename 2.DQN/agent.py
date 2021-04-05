import numpy as np
import torch


class Agent:
    def __init__(self):
        self.model = torch.load(__file__[:-8] + "/agent.pkl")
        
    def act(self, state):
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state).float()
        with torch.no_grad():
            action = self.model(state)
        return np.argmax(action.numpy())

    def reset(self):
        pass

