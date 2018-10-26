import gym
import gym_achtung

from gym.utils import play


def main():
    env = gym.make("AchtungDieKurveFullImageRandomOpponent-v1")
    play.play(env, fps=30)


if __name__ == '__main__':
    main()
