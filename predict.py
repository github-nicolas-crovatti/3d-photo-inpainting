''' Cog interface for 3D photo inpainting'''
# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

import argparse
import copy
import glob
import os
import shlex
import subprocess
import sys
import time
import gc
from functools import partial

import cv2
import imageio
import numpy as np
import scipy.misc as misc
import torch
import vispy
import yaml
from cog import BasePredictor, Input, Path
from skimage.transform import resize
from tqdm import tqdm

import MiDaS.MiDaS_utils as MiDaS_utils
from bilateral_filtering import sparse_bilateral_filtering
from boostmonodepth_utils import run_boostmonodepth
from mesh import output_3d_photo, read_ply, write_ply
from MiDaS.monodepth_net import MonoDepthNet
from MiDaS.run import run_depth
from networks import Inpaint_Color_Net, Inpaint_Depth_Net, Inpaint_Edge_Net
from utils import get_MiDaS_samples_cog, read_MiDaS_depth
import base64
import io 
from PIL import Image

cmd = "Xvfb :0 -screen 0 1024x768x24 -ac +extension GLX +render -noreset"
subprocess.Popen(shlex.split(cmd))

import os

os.environ["DISPLAY"] = ":0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

from mesh import Canvas_view



class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        os.system('mkdir -vp  /root/.cache/torch/hub/checkpoints/')
        os.system('cp -va weights/main.zip /root/.cache/torch/hub/')  
        os.system('cp -va weights/ig_resnext101_32x8-c38310e5.pth /root/.cache/torch/hub/checkpoints/')  
        pass

    def preprocessing(self, image: str, src_folder: str):
        # Convert the Data URI to a NumPy array
        image_format = image.split(";")[0].split("/")[-1]  # Extract image format (png, jpg, etc.)
        image_data = base64.b64decode(image.split(",")[-1])

        # Convert to PIL Image
        image = Image.open(io.BytesIO(image_data))

        # Specify the filename of the image
        filename = f"output_image.{image_format}"

        # Save the image
        image.save(os.path.join(src_folder, filename))

        # Return the path to the saved image
        saved_image_path = os.path.join(os.getcwd(), src_folder, filename)

        print(f"Saved image at: {saved_image_path}")
        return saved_image_path


    def predict(
        self,
        image: str = Input(description="Data URI of the input image"),
        effect: str = Input(
            description="Video animation effect", choices=["dolly-zoom-in", "zoom-in", "circle", "swing"]
        ),
    ) -> Path:



        config = yaml.load(open("argument.yml", "r"))
        # set for headless rendering
        config["offscreen_rendering"] = True
        os.makedirs(config["mesh_folder"], exist_ok=True)
        os.makedirs(config["video_folder"], exist_ok=True)
        os.makedirs(config["depth_folder"], exist_ok=True)
        
        image_path = self.preprocessing(image, config['src_folder'])

        if config["offscreen_rendering"] is True:
            vispy.use(app="egl")
        config["video_postfix"] = [effect]

        # select trajectory type, shift range based on chosen video postfix effect
        traj_types_dict = {"dolly-zoom-in": "double-straight-line",
                           'zoom-in': 'double-straight-line',
                           'circle': 'circle',
                           'swing': 'swing'}

        shift_range_dict = {"dolly-zoom-in": [[0.00], [0.00], [-0.05]],
                            "zoom-in": [[0.00], [0.00], [-0.05]],
                            "circle": [[-0.015], [-0.015], [-0.05]],
                            "swing": [[-0.015], [-0.00], [-0.05]]}

        config["traj_types"] = [traj_types_dict[effect]]
        config["x_shift_range"], config["y_shift_range"], config["z_shift_range"] = shift_range_dict[effect]

        sample_list = get_MiDaS_samples_cog(
            str(image_path), config["src_folder"], config["depth_folder"], config, config["specific"], None
        )
        normal_canvas, all_canvas = None, None

        if isinstance(config["gpu_ids"], int) and (config["gpu_ids"] >= 0):
            device = config["gpu_ids"]
            torch.cuda.empty_cache()
            gc.collect()
        else:
            device = "cpu"

        print(f"running on device {device}")

        for idx in tqdm(range(len(sample_list))):
            depth = None
            sample = sample_list[idx]
            print("Current Source ==> ", sample["src_pair_name"])
            mesh_fi = os.path.join(config["mesh_folder"], sample["src_pair_name"] + ".ply")
            image = imageio.imread(sample["ref_img_fi"])

            print(f"Running depth extraction at {time.time()}")
            if config["use_boostmonodepth"] is True:
                run_boostmonodepth(sample["ref_img_fi"], config["src_folder"], config["depth_folder"])
            elif config["require_midas"] is True:
                run_depth(
                    [sample["ref_img_fi"]],
                    config["src_folder"],
                    config["depth_folder"],
                    config["MiDaS_model_ckpt"],
                    MonoDepthNet,
                    MiDaS_utils,
                    target_w=640,
                )

            if "npy" in config["depth_format"]:
                config["output_h"], config["output_w"] = np.load(sample["depth_fi"]).shape[:2]
            else:
                config["output_h"], config["output_w"] = imageio.imread(sample["depth_fi"]).shape[:2]
            frac = config["longer_side_len"] / max(config["output_h"], config["output_w"])
            config["output_h"], config["output_w"] = int(config["output_h"] * frac), int(config["output_w"] * frac)
            config["original_h"], config["original_w"] = config["output_h"], config["output_w"]
            if image.ndim == 2:
                image = image[..., None].repeat(3, -1)
            if (
                np.sum(np.abs(image[..., 0] - image[..., 1])) == 0
                and np.sum(np.abs(image[..., 1] - image[..., 2])) == 0
            ):
                config["gray_image"] = True
            else:
                config["gray_image"] = False
            image = cv2.resize(image, (config["output_w"], config["output_h"]), interpolation=cv2.INTER_AREA)
            depth = read_MiDaS_depth(sample["depth_fi"], 3.0, config["output_h"], config["output_w"])
            mean_loc_depth = depth[depth.shape[0] // 2, depth.shape[1] // 2]
            if not (config["load_ply"] is True and os.path.exists(mesh_fi)):
                vis_photos, vis_depths = sparse_bilateral_filtering(
                    depth.copy(), image.copy(), config, num_iter=config["sparse_iter"], spdb=False
                )
                depth = vis_depths[-1]
                model = None
    
                print("Start Running 3D_Photo ...")
                print(f"Loading edge model at {time.time()}")
                depth_edge_model = Inpaint_Edge_Net(init_weights=True)
                depth_edge_weight = torch.load(config["depth_edge_model_ckpt"], map_location=torch.device(device))
                depth_edge_model.load_state_dict(depth_edge_weight)
                depth_edge_model = depth_edge_model.to(device)
                depth_edge_model.eval()

                print(f"Loading depth model at {time.time()}")
                depth_feat_model = Inpaint_Depth_Net()
                depth_feat_weight = torch.load(config["depth_feat_model_ckpt"], map_location=torch.device(device))
                depth_feat_model.load_state_dict(depth_feat_weight, strict=True)
                depth_feat_model = depth_feat_model.to(device)
                depth_feat_model.eval()
                depth_feat_model = depth_feat_model.to(device)
                print(f"Loading rgb model at {time.time()}")
                rgb_model = Inpaint_Color_Net()
                rgb_feat_weight = torch.load(config["rgb_feat_model_ckpt"], map_location=torch.device(device))
                rgb_model.load_state_dict(rgb_feat_weight)
                rgb_model.eval()
                rgb_model = rgb_model.to(device)
                graph = None

                print(f"Writing depth ply (and basically doing everything) at {time.time()}")
                rt_info = write_ply(
                    image,
                    depth,
                    sample["int_mtx"],
                    mesh_fi,
                    config,
                    rgb_model,
                    depth_edge_model,
                    depth_edge_model,
                    depth_feat_model,
                )

                if rt_info is False:
                    continue

            if config["save_ply"] is True or config["load_ply"] is True:
                verts, colors, faces, Height, Width, hFov, vFov = read_ply(mesh_fi)
            else:
                verts, colors, faces, Height, Width, hFov, vFov = rt_info

            print(f"Making video at {time.time()}")
            videos_poses, video_basename = copy.deepcopy(sample["tgts_poses"]), sample["tgt_name"]
            top = config.get("original_h") // 2 - sample["int_mtx"][1, 2] * config["output_h"]
            left = config.get("original_w") // 2 - sample["int_mtx"][0, 2] * config["output_w"]
            down, right = top + config["output_h"], left + config["output_w"]
            border = [int(xx) for xx in [top, down, left, right]]

            output_path = os.path.join(config["video_folder"], video_basename[0] + '_' + effect + '.mp4')
            normal_canvas, all_canvas = output_3d_photo(
                verts.copy(),
                colors.copy(),
                faces.copy(),
                copy.deepcopy(Height),
                copy.deepcopy(Width),
                copy.deepcopy(hFov),
                copy.deepcopy(vFov),
                copy.deepcopy(sample["tgt_pose"]),
                sample["video_postfix"],
                copy.deepcopy(sample["ref_pose"]),
                copy.deepcopy(config["video_folder"]),
                image.copy(),
                copy.deepcopy(sample["int_mtx"]),
                config,
                image,
                videos_poses,
                video_basename,
                config.get("original_h"),
                config.get("original_w"),
                border=border,
                depth=depth,
                normal_canvas=normal_canvas,
                all_canvas=all_canvas,
                mean_loc_depth=mean_loc_depth,
            )

            del rgb_model
            #del color_feat_model
            del depth_edge_model
            del depth_feat_model
            gc.collect()
            torch.cuda.empty_cache()
            print(f'Done. Saving to output path: {output_path}')
            return Path(str(output_path))
