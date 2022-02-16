#!/usr/bin/env python3


import argparse
import cv2 as cv
import gym
import gym_search


KEY_ESC = 27
KEY_RET = 13
WINDOW_SIZE = (640, 640)


parser = argparse.ArgumentParser()
parser.add_argument("env", type=str)
parser.add_argument("-d", "--delay", type=int, default=1)
parser.add_argument("-o", "--observe", action="store_true")
parser.add_argument("-v", "--verbose", action="store_true")

args = parser.parse_args()

env = gym.make(args.env)
obs = env.reset()
ep_rew = 0.0

cv.namedWindow(args.env, cv.WINDOW_AUTOSIZE)

while cv.getWindowProperty(args.env, cv.WND_PROP_VISIBLE) > 0:
    img = env.render(mode="rgb_array", observe=args.observe)
    
    img = cv.resize(img, WINDOW_SIZE, interpolation=cv.INTER_NEAREST)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    cv.imshow(args.env, img)
    key = cv.waitKey(args.delay)

    if key == KEY_ESC:
        break

    action = {
        ord(" "): env.Action.TRIGGER,
        ord("w"): env.Action.NORTH,
        ord("d"): env.Action.EAST,
        ord("s"): env.Action.SOUTH,
        ord("a"): env.Action.WEST,
    }.get(key, env.Action.NONE)
    
    obs, rew, done, info = env.step(action)
    ep_rew += rew

    if args.verbose:
        print("reward: ", rew)

    if done or key == KEY_RET:
        print("episode reward:", ep_rew)
        obs = env.reset()
        ep_rew = 0.0