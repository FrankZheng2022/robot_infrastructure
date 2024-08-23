#!/usr/bin/env python3

import argparse
import numpy as np
import cv2
from widowx_envs.widowx_env_service import WidowXClient, WidowXConfigs
import io
from pathlib import Path

def show_video(client, full_image=True):
    """
    This shows the video from the camera for a given duration.
    Full image is the image before resized to default 256x256.
    """
    res = client.get_observation()
    if res is None:
        print("No observation available... waiting")
        return None, None
    if full_image:
        org_img = res["full_image"]
        img = cv2.cvtColor(org_img, cv2.COLOR_RGB2BGR)
    else:
        org_img = res["image"]
        img = (org_img.reshape(3, 256, 256).transpose(1, 2, 0) * 255).astype(np.uint8)
    cv2.imshow("Robot Camera", img)
    #cv2.waitKey(20)  # 20 ms
    return org_img, img

def save_episode(episode, fn):
    with io.BytesIO() as bs:
        np.savez_compressed(bs, **episode)
        bs.seek(0)
        with fn.open('wb') as f:
            f.write(bs.read())

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

def main():
    parser = argparse.ArgumentParser(description='Teleoperation for WidowX Robot')
    parser.add_argument('--id', type=int, default=0)
    parser.add_argument('--ip', type=str, default='localhost')
    parser.add_argument('--port', type=int, default=5556)
    parser.add_argument('--save_dir', type=str, default='/home/ruzheng/bridge_data_robot/trainingdata/test')
    args = parser.parse_args()

    client = WidowXClient(host=args.ip, port=args.port)
    client.init(WidowXConfigs.DefaultEnvParams, image_size=256)
    save_path = Path(args.save_dir)

    print_help()
    cv2.namedWindow("Robot Camera")
    is_open = 1
    running = True
    import time
    time.sleep(10)
    actions, obses = [], [show_video(client)[0]]
    while running:
        # Check for key press
        key = cv2.waitKey(100) & 0xFF

        # escape key to quit
        if key == ord('q'):
            print("Quitting teleoperation.")
            running = False
            continue
        # Handle key press for robot control
        # translation
        action = None
        
        if key == ord('w'):
            action = np.array([0.02, 0, 0, 0, 0, 0, is_open])
        elif key == ord('s'):
            action = np.array([-0.02, 0, 0, 0, 0, 0, is_open])
        elif key == ord('a'):
            action = np.array([0, 0.02, 0, 0, 0, 0, is_open])
        elif key == ord('d'):
            action = np.array([0, -0.02, 0, 0, 0, 0, is_open])
        elif key == ord('z'):
            action = np.array([0, 0, 0.02, 0, 0, 0, is_open])
        elif key == ord('c'):
            action = np.array([0, 0, -0.02, 0, 0, 0, is_open])
        
        # rotation
        elif key == ord('i'):
            action = np.array([0, 0, 0, 0.04, 0, 0, is_open])
        elif key == ord('k'):
            action = np.array([0, 0, 0, -0.04, 0, 0, is_open])
        elif key == ord('j'):
            action = np.array([0, 0, 0, 0, 0.04, 0, is_open])
        elif key == ord('l'):
            action = np.array([0, 0, 0, 0, -0.04, 0, is_open])
        elif key == ord('n'):
            action = np.array([0, 0, 0, 0, 0, 0.04, is_open])
        elif key == ord('m'):
            action = np.array([0, 0, 0, 0, 0, -0.04, is_open])   
        
        # space bar to change gripper state
        elif key == ord('x'):
            is_open = 1 - is_open
            print("Gripper is now: ", is_open)
            action = np.array([0, 0, 0, 0, 0, 0, is_open])
        elif key == ord('r'):
            print("Resetting robot...")
            client.reset()
            print_help()
        
        img, _ = show_video(client)
        if action is not None and img is not None:
            client.step_action(action, blocking=True)
            actions.append(action)
            obses.append(img)

    client.stop()  # Properly stop the client
    cv2.destroyAllWindows()
    print("Teleoperation ended.")

    from PIL import Image
    pil_images = [Image.fromarray(image) for image in obses]
    # Save as GIF
    output_path = save_path / f"output_{args.id}.gif"
    pil_images[0].save(output_path, save_all=True, append_images=pil_images[1:], loop=0, duration=500)
    episode_fn = save_path / f'traj_{args.id}.npz'
    episode = {"obs": obses, "action":actions}
    save_episode(episode, episode_fn)

if __name__ == "__main__":

    main()
