import gym
import gym_achtung
from gym.wrappers import Monitor
import pickle
import os
import time
import matplotlib.pyplot as plt


# The world's simplest agent!
class RandomAgent(object):
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation, reward, done):
        return self.action_space.sample()


def main():
    env = gym.make("AchtungDieKurve-v1")
    agent = RandomAgent(env.action_space)

    timestr = time.strftime("%Y%m%d-%H%M%S")
    outputPath = './Sharks/' + timestr
    os.makedirs(outputPath)

    #env = Monitor(env, directory= '/Monitor', force=True)
    render = False

    meanRewards = []
    numberOfEvaluations = 100
    eval = 0
    while eval < numberOfEvaluations:
        eval += 1
        obs, done = env.reset(), False
        episode_rew = 0
        rew = 0
        while not done:
            if render:
                env.render()

            action = agent.act(obs, rew, done)


            obs, rew, done, _ = env.step([action])
            #time.sleep(0.1)

            episode_rew += rew

        print("Episode reward", episode_rew)
        meanRewards.append(episode_rew)

    outputNameReward = outputPath + '/EnjoyRandomReward.pkl'

    with open(outputNameReward, 'wb') as f:
        pickle.dump(meanRewards, f)
        print('Rewards dumped @ ' + outputNameReward )

if __name__ == '__main__':
    main()
