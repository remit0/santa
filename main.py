from model import DQN, ReplayMemory
from environment import Environment


def init_memory(env, memory, size):
    while len(memory) < size:
        state = env.reset()
        done = False
        while not done:
            action = env.action_space.sample()
            next_state, reward, done, _ = env.step(action)
            memory.push(state, action, next_state, reward, done)
            state = next_state


def compute_eps(n_episode, threshold=5000):
    if n_episode > threshold:
        return 0.1
    else:
        return 1 - ((0.9 * n_episode) / threshold)


def train_one_episode(env, dqn, memory, gamma, batch_size, eps, period, iters):
    state = env.reset()
    done = False
    episode_reward = 0

    while not done:

        action = dqn.sample_action(state, eps)
        next_state, reward, done, _ = env.step(action)
        memory.push(state, action, next_state, reward, done)
        dqn.update(memory, batch_size, gamma)

        if iters % period == 0:
            dqn.update_target_net()

        iters += 1
        episode_reward += reward
        state = next_state

    return episode_reward, iters


def make_submission(env, dqn, filename="output.csv"):
    import pandas as pd

    state = env.reset()
    done = False

    indices = []
    days = []

    while not done:
        family_index = env.family_index
        action = dqn.sample_action(state, 0)
        family_day = env.get_day_from_action(action)
        next_state, _, done, _ = env.step(action)

        indices.append(family_index)
        days.append(family_day)

    submission = pd.DataFrame({"family_id": indices, "assigned_day": days})
    submission.to_csv(filename, index=False)


def main():
    env = Environment()

    gamma = 0.99
    period = 100
    learning_rate = 1e-6
    min_memory_size = 1000
    max_memory_size = 10000
    batch_size = 32
    num_episodes = 100
    layers = [env.observation_space.n, 444, 222, env.action_space.n]

    memory = ReplayMemory(max_memory_size)
    dqn = DQN(layers, learning_rate)
    init_memory(env, memory, min_memory_size)

    iters = 0
    for n_ep in range(num_episodes):
        eps = compute_eps(n_ep, 10)
        reward, iters = train_one_episode(env, dqn, memory, gamma, batch_size, eps, period, iters)
        print(n_ep, reward)

    make_submission(env, dqn)


if __name__ == "__main__":
    main()
