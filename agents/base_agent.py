# agents/base_agent.py

class BaseAgent:
    def __init__(self, name):

        self.name = name

    def run(self, state):

        return state