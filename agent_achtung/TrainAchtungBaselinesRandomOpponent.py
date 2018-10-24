import gym_achtung
import gym
#from gym.wrappers import Monitor
from baselines import deepq
import time
import json

def main():
    env = gym.make("AchtungDieKurveRandomOpponent-v1")

    #env = Monitor(env, directory='./Monitor', force=True)

    timestr = time.strftime("%Y%m%d-%H%M%S")
    outputPathModel = 'achtung_RO_model_' + str(timestr) + '.pkl'
    outputPathInfo = 'achtung_RO_info_' + str(timestr) + '.txt'

    infoDict = {}
    infoDict['total_timesteps'] = 1000000
    infoDict['lr'] = 1e-4
    infoDict['buffer_size'] = 100000
    infoDict['exploration_fraction'] = 0.2
    infoDict['prioritized_replay'] = True

    print("Saving training information to achtung_RO_info_%Y%m%d-%H%M%S.txt")
    with open(outputPathInfo, 'w') as file:
        file.write(json.dumps(infoDict))  # use `json.loads` to do the reverse


    act = deepq.learn(
        env,
        network='mlp',
        lr=5e-4,
        total_timesteps=infoDict['total_timesteps'],
        buffer_size=infoDict['buffer_size'],
        exploration_fraction=infoDict['exploration_fraction'],
        exploration_final_eps=0.02,
        prioritized_replay=infoDict['prioritized_replay'],
        print_freq=1000,
        callback=None,
        render=False
    )
    print("Saving model to achtung_RO_model_%Y%m%d-%H%M%S.pkl")
    act.save(outputPathModel)




if __name__ == '__main__':
    main()