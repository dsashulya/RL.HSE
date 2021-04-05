import random
import numpy as np
import os
import torch
from .train import Actor


class Agent:
    def __init__(self):
        self.model = torch.load(__file__[:-8] + "/agent.pkl")
        
    def act(self, state):
        with torch.no_grad():
            state = torch.tensor(np.array(state))
            action = self.model.forward(state)
        return action

    def reset(self):
        pass

