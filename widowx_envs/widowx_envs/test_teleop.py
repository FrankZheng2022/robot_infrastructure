#!/usr/bin/env python3

import argparse
import numpy as np
import cv2
from widowx_envs.widowx_env_service import WidowXClient, WidowXConfigs
import threading
import pyspacemouse
from typing import Tuple
import time
from pathlib import Path
import io


class SpaceMouseExpert:
    """
    This class provides an interface to the SpaceMouse.
    It continuously reads the SpaceMouse state and provide
    a "get_action" method to get the latest action and button state.
    """

    def __init__(self):
        pyspacemouse.open()

        self.state_lock = threading.Lock()
        self.latest_data = {"action": np.zeros(6), "buttons": [0, 0]}
        # Start a thread to continuously read the SpaceMouse state
        self.thread = threading.Thread(target=self._read_spacemouse)
        self.thread.daemon = True
        self.thread.start()

    def _read_spacemouse(self):
        while True:
            state = pyspacemouse.read()
            with self.state_lock:
                self.latest_data["action"] = np.array(
                    #[-state.y, state.x, state.z, -state.roll, -state.pitch, -state.yaw]
                    [state.y, -state.x, state.z, state.roll, state.pitch, -state.yaw]
                )  # spacemouse axis matched with robot base frame
                self.latest_data["buttons"] = state.buttons

    def get_action(self) -> Tuple[np.ndarray, list]:
        """Returns the latest action and button state of the SpaceMouse."""
        with self.state_lock:
            return self.latest_data["action"], self.latest_data["buttons"]


def show_video(client, full_image=True):
    """
    This shows the video from the camera for a given duration.
    Full image is the image before resized to default 256x256.
    """
    res = client.get_observation()
    if res is None:
        print("No observation available... waiting")
        return None, None, None
    if full_image:
        org_img = res["full_image"]
        img = cv2.cvtColor(org_img, cv2.COLOR_RGB2BGR)
    else:
        org_img = res["image"]
        img = (org_img.reshape(3, 256, 256).transpose(1, 2, 0) * 255).astype(np.uint8)
    cv2.imshow("Robot Camera", img)
    state = res["state"]
    #cv2.waitKey(20)  # 20 ms
    return org_img, img, state

def images_to_mp4(image_list, output_path, fps=10):
    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    height, width, layers = 480,640,3
    video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for img in image_list:
        bgr_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        video.write(np.array(bgr_img, dtype=np.uint8))

    video.release()

def save_episode(episode, fn):
    with io.BytesIO() as bs:
        np.savez_compressed(bs, **episode)
        bs.seek(0)
        with fn.open('wb') as f:
            f.write(bs.read())

def load_episode(fn):
    with fn.open('rb') as f:
        episode = np.load(f)
        episode = {k: episode[k] for k in episode.keys()}
        return episode


print_yellow = lambda x: print("\033[93m {}\033[00m" .format(x))

def print_help():
    print_yellow("  Teleop Controls:")
    print_yellow("    w, s : move forward/backward")
    print_yellow("    a, d : move left/right")
    print_yellow("    z, c : move up/down")
    print_yellow("    i, k:  rotate yaw")
    print_yellow("    j, l:  rotate pitch")
    print_yellow("    n  m:  rotate roll")
    print_yellow("    space: toggle gripper")
    print_yellow("    r: reset robot")
    print_yellow("    q: quit")

def scale(action, scale_mov=0.03, scale_rot=0.2):
    action_mov = action[:3]*scale_mov
    action_rot = action[3:6]*scale_rot
    return np.hstack([action_mov, action_rot])


def main():
    parser = argparse.ArgumentParser(description='Teleoperation for WidowX Robot')
    parser.add_argument('--ip', type=str, default='localhost')
    parser.add_argument('--port', type=int, default=5556)
    parser.add_argument('--save_dir', type=str, default='/home/ruzheng/bridge_data_robot/trainingdata/pickplace_corn')
    parser.add_argument('--id', type=int, default=0)
    args = parser.parse_args()

    client = WidowXClient(host=args.ip, port=args.port)
    client.init(WidowXConfigs.DefaultEnvParams, image_size=256)
    
    episode = load_episode(Path('/home/ruzheng/bridge_data_robot/trainingdata/test/traj_0.npz'))

    cv2.namedWindow("Robot Camera")
    is_open, i = 1, 0
    running = True
    time.sleep(5)
    spacemouse = SpaceMouseExpert()
    img, _, state = show_video(client)
    actions, obses, states = [], [img], [state]
    while running:        
        with np.printoptions(precision=3, suppress=True):
            action, buttons = spacemouse.get_action()
            # escape key to quit
            if buttons[8] > 0.:
                print("Quitting teleoperation.")
                running = False
                continue
            #print(f"Spacemouse action: {action}, buttons: {buttons}")
            if buttons[7] > 0.:
                print("Resetting robot...")
                client.reset()
            # if np.sum(np.abs(action)) > 0 or buttons[5] > 0:
            #     if buttons[5] > 0.:
            #         is_open = 1 - is_open
            #         #print('close/open the gripper')
            #     action = scale(action, scale_mov=0.02, scale_rot=0.1)
            #     action = np.hstack([action, is_open])
            action = episode['action'][i]
            i += 1
            client.step_action(action)
            img, _, state = show_video(client)
            show_video(client)
            


    client.stop()  # Properly stop the client
    #cv2.destroyAllWindows()
    print("Teleoperation ended.")

if __name__ == "__main__":
    main()
