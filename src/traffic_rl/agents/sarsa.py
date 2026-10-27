from traffic_rl.agents.q_learning import QLearningAgent


class SarsaAgent(QLearningAgent):
    def bootstrap_value(self, next_state: int, next_action: int) -> float:
        return float(self.q[next_state, next_action])
