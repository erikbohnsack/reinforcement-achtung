import gym
import gym_achtung
from baselines import deepq
from gym.wrappers import Monitor
import pickle
import os
import time
import matplotlib.pyplot as plt


modelToRun = "achtung_FullImage_model_20181024-162834.pkl"

def main():
    env = gym.make("AchtungDieKurveFullImage-v1")
    act = deepq.learn(env, network='conv_only', total_timesteps=0, load_path=modelToRun)

    timestr = time.strftime("%Y%m%d-%H%M%S")
    outputPath = './Sharks/' + timestr
    os.makedirs(outputPath)

    #env = Monitor(env, directory= '/Monitor', force=True)

    meanRewards = []
    qValues = []
    numberOfEvaluations = 50
    eval = 0
    while eval < numberOfEvaluations:
        eval += 1
        obs, done = env.reset(), False
        episode_rew = 0
        episode_qVal = []
        while not done:
            env.render()

            getActionQvalue = act(obs)
            action = getActionQvalue[0][0]

            obs, rew, done, _ = env.step(action)
            #time.sleep(0.1)

            episode_rew += rew
            episode_qVal.append(getActionQvalue[1])

        print("Episode reward", episode_rew)
        meanRewards.append(episode_rew)
        qValues.append(episode_qVal)


    outputNameReward = outputPath + '/EnjoyReward.pkl'
    outputNameQvalues = outputPath + '/EnjoyQvalues.pkl'

    with open(outputNameReward, 'wb') as f:
        pickle.dump(meanRewards, f)
        print('Rewards dumped @ ' + outputNameReward )

    with open(outputNameQvalues, 'wb') as f:
        pickle.dump(qValues, f)
        print('Qvalues dumped @ ' + outputNameQvalues)


if __name__ == '__main__':
    main()
