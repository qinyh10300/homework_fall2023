from cs285.agents.pg_agent import PGAgent
import numpy as np

agent = PGAgent(
        ob_dim=1,
        ac_dim=1,
        discrete=1,
        n_layers=1,
        layer_size=1,
        gamma=0.9,
        learning_rate=1,
        use_baseline=1,
        use_reward_to_go=1,
        normalize_advantages=1,
        baseline_learning_rate=1,
        baseline_gradient_steps=1,
        gae_lambda=1,
    )

rewards = [np.array([1.0,2.0,3.0]), np.array([10.0,5.0])]

print(agent._discounted_reward_to_go(rewards))