import os
import json
import argparse

def txt2json(root_path):
    camera_path = os.path.join(root_path, 'camera.txt')
    save_path = os.path.join(root_path, 'camera_extrinsics.json')

    f = open(camera_path, "r+")

    lines = f.readlines()

    cameras = {}

# import IPython; IPython.embed(); exit()

    for i in range(0, len(lines), 14):
        filename = lines[i+1][:-1]

        cameras[filename] = dict(
        focal_length = list(map(float,lines[i+3][:-1].split(' '))),
        image_center = list(map(float,lines[i+4][:-1].split(' '))),
        translation = list(map(float,lines[i+5][:-2].split(' '))),
        camera_pos = list(map(float,lines[i+6][:-2].split(' '))),
        quaternion = list(map(float,lines[i+8][:-2].split(' '))),
        rotation = [list(map(float, lines[i+9][:-2].split(' '))), 
                list(map(float, lines[i+10][:-2].split(' '))), 
                list(map(float, lines[i+11][:-2].split(' ')))],
    )

    json_str = json.dumps(cameras, indent=4)

    with open(save_path, 'w') as json_file:
        json_file.write(json_str)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path',type=str, default = './')
    args = parser.parse_args()
    txt2json(args.path)
