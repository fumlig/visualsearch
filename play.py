#!/usr/bin/env python3


import argparse
import cv2 as cv
import gym
import gym_ptz


KEY_ESC = 27
KEY_RET = 13
WINDOW_NAME = "play"
WINDOW_SIZE = (640, 640)


parser = argparse.ArgumentParser()
parser.add_argument("env", type=str)
parser.add_argument("-d", "--delay", type=int, default=1)
parser.add_argument("-o", "--observe", action="store_true")
args = parser.parse_args()

env = gym.make(args.env)
obs = env.reset()

cv.namedWindow(WINDOW_NAME, cv.WINDOW_AUTOSIZE)

while cv.getWindowProperty(WINDOW_NAME, cv.WND_PROP_VISIBLE) > 0:
    img = env.render(mode="rgb_array", observe=args.observe)
    
    cv.imshow(WINDOW_NAME, cv.cvtColor(cv.resize(img, WINDOW_SIZE, interpolation=cv.INTER_NEAREST), cv.COLOR_BGR2RGB))
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
    
    obs, reward, done, _info = env.step(action)

    if done or key == KEY_RET:
        obs = env.reset()