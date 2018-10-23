import gym
import gym_achtung
from baselines import deepq
import matplotlib.pyplot as plt

modelToRun = '/Users/adamlilja/Documents/Skola/deep-machine-learning/project/achtung-die-PLE/agent_achtung/achtung_model.pkl'

def main():
    env = gym.make("AchtungDieKurve-v1")
    act = deepq.learn(env, network='mlp', total_timesteps=0, load_path=modelToRun)

    meanRewards = []

    numberOfEvaluations = 25
    eval = 0
    while eval < numberOfEvaluations:
        eval += 1
        obs, done = env.reset(), False
        episode_rew = 0
        while not done:
            env.render()
            obs, rew, done, _ = env.step(act(obs)[0])
            episode_rew += rew

        print("Episode reward", episode_rew)
        meanRewards.append(episode_rew)

    plt.plot(meanRewards)
    plt.ylabel('Mean Reward')
    plt.xlabel('Episode')
    plt.show()

if __name__ == '__main__':
    main()