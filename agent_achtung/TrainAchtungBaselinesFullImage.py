import gym_achtung
import gym
#from gym.wrappers import Monitor
from baselines import deepq
import time
import json

def main():
    env = gym.make("AchtungDieKurveFullImage-v1")

    #env = Monitor(env, directory='./Monitor', force=True)

    timestr = time.strftime("%Y%m%d-%H%M%S")
    outputPathModel = 'achtung_FullImage_model_' + str(timestr) + '.pkl'
    outputPathInfo = 'achtung_FullImage_info_' + str(timestr) + '.txt'

    infoDict = {}
    infoDict['total_timesteps'] = 1000000
    infoDict['lr'] = 1e-4
    infoDict['buffer_size'] = 100000
    infoDict['exploration_fraction'] = 0.2
    infoDict['prioritized_replay'] = True

    print("Saving training information to achtung_FullImage_info_%Y%m%d-%H%M%S.txt")
    with open(outputPathInfo, 'w') as file:
        file.write(json.dumps(infoDict))  # use `json.loads` to do the reverse

    act = deepq.learn(
        env,
        network='conv_only',
        lr=1e-4,
        buffer_size=10000,
        exploration_fraction=0.1,
        exploration_final_eps=0.01,
        train_freq=4,
        learning_starts=10000,
        target_network_update_freq=1000,
        gamma=0.99,
        prioritized_replay=True,
        prioritized_replay_alpha=0.6,
        checkpoint_freq=10000,
        checkpoint_path=None,
        dueling=True,
        render=False
    )
    print("Saving model to achtung_FullImage_model_%Y%m%d-%H%M%S.pkl")
    act.save(outputPathModel)




if __name__ == '__main__':
    main()