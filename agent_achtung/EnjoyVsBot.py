import gym
import gym_achtung
from baselines import deepq
import time

from gym.utils import play

model_to_run = "achtung_best_bot.pkl"

def main():
    env = gym.make("AchtungDieKurveAgainstBot-v1")
    number_of_evaluations = 50
    eval = 0
    act = deepq.learn(env, network='mlp', total_timesteps=0, load_path=model_to_run)
    while eval < number_of_evaluations:
        eval += 1
        obs, done = env.reset(), False
        print(eval)
        while not done:
            time.sleep(0.01)
            env.render()
            getActionQvalue = act(obs)
            action = getActionQvalue[0][0]
            obs, rew, done, _ = env.step(action)


if __name__ == '__main__':
    main()
