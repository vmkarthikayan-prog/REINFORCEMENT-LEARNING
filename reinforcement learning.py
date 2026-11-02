import numpy as np
import matplotlib.pyplot as plt

# Problem Parameters
STATES = [1, 2, 3, 4, 5]  # Indexing: 1=A, 2=B, 3=C, 4=D, 5=E
START_STATE = 3           # State C
TRUE_VALUES = np.array([0, 1/6, 2/6, 3/6, 4/6, 5/6, 0])  # Includes terminal states 0 and 6

def generate_episode():
    """Generates a trajectory of (states, rewards)."""
    trajectory = []
    state = START_STATE
    trajectory.append(state)
    
    while state in STATES:
        # Move left (-1) or right (+1) with probability 0.5
        action = -1 if np.random.rand() < 0.5 else 1
        next_state = state + action
        
        # Reward is 1 only if terminating on the right (state 6)
        reward = 1.0 if next_state == 6 else 0.0
        
        trajectory.append((reward, next_state))
        state = next_state
        
    return trajectory

def run_td0(alpha, num_episodes, V_init=None):
    """Runs tabular TD(0) for a given number of episodes."""
    V = np.array([0.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.0]) if V_init is None else V_init.copy()
    
    for _ in range(num_episodes):
        state = START_STATE
        while state in STATES:
            action = -1 if np.random.rand() < 0.5 else 1
            next_state = state + action
            reward = 1.0 if next_state == 6 else 0.0
            
            # TD(0) Online Update
            V[state] += alpha * (reward + V[next_state] - V[state])
            state = next_state
            
    return V[1:6]  # Return non-terminal state estimates

def run_constant_alpha_mc(alpha, num_episodes):
    """Runs constant-alpha every-visit Monte Carlo for a given number of episodes."""
    V = np.array([0.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.0])
    
    for _ in range(num_episodes):
        trajectory = generate_episode()
        
        # Terminal reward is the final reward in the trajectory
        final_reward = trajectory[-1][0]
        G = final_reward  # Undiscounted return gamma = 1
        
        # Extract states visited in the episode (excluding the terminal state)
        visited_states = [trajectory[0]] + [step[1] for step in trajectory[1:-1]]
        
        # MC Offline Update at episode end
        for s in visited_states:
            V[s] += alpha * (G - V[s])
            
    return V[1:6]

# --- Figure 6.6 Replication: State Values vs Episodes for TD(0) ---
def plot_figure_6_6():
    episodes_to_plot = [0, 1, 10, 100]
    alpha = 0.1
    
    plt.figure(figsize=(8, 5))
    plt.plot(['A', 'B', 'C', 'D', 'E'], TRUE_VALUES[1:6], 'k-o', label='True Values')
    
    # Run independent simulations up to each episode milestone
    for ep in episodes_to_plot:
        V_est = run_td0(alpha, ep)
        plt.plot(['A', 'B', 'C', 'D', 'E'], V_est, '-o', label=f'{ep} episodes')
        
    plt.xlabel('State')
    plt.ylabel('Estimated Value')
    plt.title('TD(0) State Value Estimates Over Episodes (Fig 6.6)')
    plt.legend()
    plt.grid(True)
    plt.show()

# --- Figure 6.7 Replication: Empirical Comparison of RMSE ---
def compute_rmse(V_est):
    return np.sqrt(np.mean((V_est - TRUE_VALUES[1:6]) ** 2))

def plot_figure_6_7(runs=100, max_episodes=100):
    td_alphas = [0.15, 0.1, 0.05]
    mc_alphas = [0.01, 0.02, 0.03, 0.04]
    
    plt.figure(figsize=(9, 6))
    
    # Evaluate TD(0)
    for alpha in td_alphas:
        errors = np.zeros(max_episodes + 1)
        for _ in range(runs):
            V = np.array([0.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.0])
            errors[0] += compute_rmse(V[1:6])
            for ep in range(1, max_episodes + 1):
                state = START_STATE
                while state in STATES:
                    action = -1 if np.random.rand() < 0.5 else 1
                    next_state = state + action
                    reward = 1.0 if next_state == 6 else 0.0
                    V[state] += alpha * (reward + V[next_state] - V[state])
                    state = next_state
                errors[ep] += compute_rmse(V[1:6])
        errors /= runs
        plt.plot(range(max_episodes + 1), errors, label=f'TD alpha={alpha}')
        
    # Evaluate Constant-Alpha MC
    for alpha in mc_alphas:
        errors = np.zeros(max_episodes + 1)
        for _ in range(runs):
            V = np.array([0.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.0])
            errors[0] += compute_rmse(V[1:6])
            for ep in range(1, max_episodes + 1):
                trajectory = generate_episode()
                G = trajectory[-1][0]
                visited_states = [trajectory[0]] + [step[1] for step in trajectory[1:-1]]
                for s in visited_states:
                    V[s] += alpha * (G - V[s])
                errors[ep] += compute_rmse(V[1:6])
        errors /= runs
        plt.plot(range(max_episodes + 1), errors, '--', label=f'MC alpha={alpha}')
        
    plt.xlabel('Episodes')
    plt.ylabel('Empirical RMS Error')
    plt.title('Empirical Comparison of TD(0) and Constant-alpha MC (Fig 6.7)')
    plt.legend()
    plt.grid(True)
    plt.show()

# Run plots
plot_figure_6_6()
plot_figure_6_7(runs=50, max_episodes=100)
