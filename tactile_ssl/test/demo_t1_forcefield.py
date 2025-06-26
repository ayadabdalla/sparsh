# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC-BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

import io
import os
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt

from typing import Optional
from tactile_ssl import algorithm

import torch
import torch.nn.functional as F
from tqdm import tqdm
from scipy import stats
from PIL import Image
from collections import deque
from omegaconf import OmegaConf

from .test_task import TestTaskSL

from tactile_ssl.data.vision_based_interactive import DemoForceFieldData
from tactile_ssl.data.digit.utils import compute_diff

import xmlrpc.client


class RPCClient:
    def __init__(self, server_url="http://localhost:8079/RPC2"):
        self.server_url = server_url
        self.proxy = xmlrpc.client.ServerProxy(self.server_url, allow_none=True)

    def call(self, method_name, *args):
        """Call a remote method by name with provided arguments"""
        if not hasattr(self.proxy, method_name):
            raise AttributeError(f"Remote server has no method named '{method_name}'")

        method = getattr(self.proxy, method_name)
        try:
            result = method(*args)
            return result
        except xmlrpc.client.Fault as err:
            print(f"A fault occurred: {err.faultCode} {err.faultString}")
            raise
        except xmlrpc.client.ProtocolError as err:
            print(f"Protocol error: {err.url} {err.errcode} {err.errmsg}")
            raise
        except Exception as err:
            print(f"Unexpected error: {err}")
            raise

    def close(self):
        """Close the connection to the server"""
        if hasattr(self.proxy, "_ServerProxy__close"):
            self.proxy._ServerProxy__close()


class DemoForceField(TestTaskSL):
    def __init__(
        self,
        digit_serial: Optional[str],
        gelsight_device_id: Optional[int],
        device,
        module: algorithm.Module,
        digit_sensor = None,
        robot = None,
        force_controller_parameters: Optional[dict] = None,
    ):
        super().__init__(
            device=device,
            module=module,
        )
        self.robot=robot
        self.digit_sensor = digit_sensor
        self.digit_serial = digit_serial
        self.force_controller_parameters = force_controller_parameters
        self.gelsight_device_id = gelsight_device_id
        print("initializing RPC client...")
        # self.robot = RPCClient("http://172.29.4.15:8079/RPC2")
        # self.robot.call("set_guiding_mode", True)

        self.init()

    def init(self):
        self.config = OmegaConf.load(
            f"/home/epon04yc/sparsh/outputs_sparsh/config.yaml"
        )
        self.sensor = self.config.sensor
        # self.sensor = "digit"
        self.th_no_contact = 0.017 if self.sensor == "digit" else 0.0198
        self.sensor_handler = DemoForceFieldData(
            config=self.config.data.dataset.config,
            digit_serial=self.digit_serial,
            gelsight_device_id=self.gelsight_device_id,
            digit_sensor=self.digit_sensor,
        )
        self.img_buffer = deque(maxlen=5)
        self._set_bg_template()

    def _adjust_gripper(self, delta: float):
        """Adjust gripper width by delta amount."""
        try:
            current_width = self.robot.call("get_gripper_width")["gripper_width"]
            new_width = float(np.clip(current_width + delta, 0, 1))
            result = self.robot.call("set_gripper_width", new_width)
            print(f"Gripper width: {current_width:.2f} -> {new_width:.2f}")
            print(result)
        except Exception as e:
            print(f"Error adjusting gripper: {e}")

    def _normalize_image(self, x):
        """Rescale image pixels to span range [0, 1]"""
        ma = float(x.max().cpu().data)
        mi = float(x.min().cpu().data)
        d = ma - mi if ma != mi else 1e5
        return torch.clip((x - mi) / d, 0.0, 1.0)

    def _normal2mask(self, heightmap, bg_template, b, r, clip):
        heightmap = heightmap.squeeze()
        heightmap = heightmap[b:-b, b:-b]

        heightmap = heightmap * 255
        bg_template = bg_template * 255

        diff_heights = heightmap
        diff_heights[diff_heights < clip] = 0
        threshold = torch.quantile(diff_heights, 0.9) * r
        threshold = torch.clip(threshold, 0, 240)
        contact_mask = diff_heights > threshold

        padded_contact_mask = torch.zeros_like(bg_template, dtype=bool)
        padded_contact_mask[b:-b, b:-b] = contact_mask

        return padded_contact_mask

    def _set_bg_template(self):
        bg = self.sensor_handler.bg
        bg = compute_diff(bg, bg)
        bg = Image.fromarray(bg)
        bg = self.sensor_handler.transform_resize(bg).unsqueeze(0).to(self.device)
        bg = torch.cat([bg, bg], dim=1)
        outputs_forces = self.module(bg)
        self.bg_template = self._normalize_image(outputs_forces["normal"]).squeeze()

    def _init_shear(self, shear, normal, margin=0, spacing=12):
        self.fig, self.ax = plt.subplots()
        self.fig.patch.set_facecolor("black")
        h, w, *_ = shear.shape

        nx = int((w - 2 * margin) / spacing)
        ny = int((h - 2 * margin) / spacing)

        x = np.linspace(margin, w - margin - 1, nx, dtype=np.int64)
        y = np.linspace(margin, h - margin - 1, ny, dtype=np.int64)

        shear = shear[np.ix_(y, x)]
        u = shear[:, :, 0]
        v = shear[:, :, 1]
        m = normal[np.ix_(y, x)]

        rad_max = 20.0
        epsilon = 1e-5
        u = u / (rad_max + epsilon)
        v = v / (rad_max + epsilon)

        u = np.clip(u, -1.0, 1.0)
        v = np.clip(v, -1.0, 1.0)
        uu = u.copy()
        vv = v.copy()
        r = np.sqrt(u**2 + v**2)
        idx_clip = np.where(r < 0.01)
        uu[idx_clip] = 0.0
        vv[idx_clip] = 0.0

        uu = uu / (np.abs(uu).max() + epsilon)
        vv = vv / (np.abs(vv).max() + epsilon)

        kwargs = {
            **dict(
                angles="uv",
                scale_units="dots",
                scale=0.025,
                width=0.007,
                cmap="inferno",
                edgecolor="face",
            ),
        }
        self.ax_shear = self.ax.quiver(y, x, uu, -vv, m, **kwargs)
        self.ax.set_ylim(sorted(self.ax.get_ylim(), reverse=True))
        self.ax.set_facecolor("black")
        self.ax.set_xticks([])
        self.ax.set_yticks([])

        # remove white border
        self.ax.spines["top"].set_visible(False)
        self.ax.spines["right"].set_visible(False)
        self.ax.spines["bottom"].set_visible(False)
        self.ax.spines["left"].set_visible(False)

        cbar = self.fig.colorbar(self.ax_shear, ax=self.ax, orientation="vertical")
        cbar.set_label(
            "Normalized normal force", labelpad=0.05, fontsize=10, color="white"
        )
        cbar.set_ticks([0, 1])
        cbar.ax.tick_params(labelcolor="white", labelsize=10)

    def update_shear(self, shear, normal, margin=0, spacing=12):
        h, w, *_ = shear.shape

        nx = int((w - 2 * margin) / spacing)
        ny = int((h - 2 * margin) / spacing)

        x = np.linspace(margin, w - margin - 1, nx, dtype=np.int64)
        y = np.linspace(margin, h - margin - 1, ny, dtype=np.int64)

        shear = shear[np.ix_(y, x)]
        u = shear[:, :, 0]
        v = shear[:, :, 1]
        m = normal[np.ix_(y, x)]

        rad_max = 20.0
        epsilon = 1e-5
        u = u / (rad_max + epsilon)
        v = v / (rad_max + epsilon)

        u = np.clip(u, -1.0, 1.0)
        v = np.clip(v, -1.0, 1.0)
        uu = u.copy()
        vv = v.copy()
        r = np.sqrt(u**2 + v**2)
        idx_clip = np.where(r < 0.01)
        uu[idx_clip] = 0.0
        vv[idx_clip] = 0.0

        uu = uu / (np.abs(uu).max() + epsilon)
        vv = vv / (np.abs(vv).max() + epsilon)

        self.ax_shear.set_UVC(uu, -vv, m)
        self.ax_shear.set_clim(0.0, 1.0)

        self.fig.canvas.draw()

        with io.BytesIO() as buff:
            self.fig.savefig(
                buff,
                format="png",
                bbox_inches="tight",
                pad_inches=0,
                transparent=False,
                dpi=150,
            )
            buff.seek(0)
            img = Image.open(io.BytesIO(buff.read()))
        return np.array(img)

    def run_model(self):
        self.robot.call("set_gripper_width", 1)
        self.robot.call("set_guiding_mode", True)
        border, ratio, clip = 15, 1.0, 50
        prev_error = 0.0
        prev_shear_error = 0.0
        prev_shear_error_max = 0.0
        buffer = []
        shear_buffer = []
        shear_buffer_max = []
        tactile_images = []
        # cv2.namedWindow("sparsh", cv2.WINDOW_NORMAL)
        init_done = False

        print("Sparsh demo starting. Press 'q' to exit the demo. \n")

        print("Starting range calibration ...")
        print("Please do not touch the sensor.")
        if self.digit_sensor is not None:
            self.sensor_handler.touch_sensor = self.digit_sensor

        for _ in range(5):
            sample = self.sensor_handler.get_model_inputs()
            img_fg = sample["image"][0:3].permute(1, 2, 0).cpu().numpy()
            img_fg = cv2.GaussianBlur(img_fg, (5, 5), 0)
            self.img_buffer.append(img_fg)
            time.sleep(0.1)

        # compute average image and std image
        avg_img_no_contact = np.mean(np.array(self.img_buffer), axis=0)
        std_img_no_contact = np.std(avg_img_no_contact) * 1.7
        counter = 0

        while True:
            sample = self.sensor_handler.get_model_inputs()
            current_tactile_image = sample["current_image_color"]
            tactile_images.append(current_tactile_image)

            outputs_forces = {"normal": None, "shear": None}

            # forward pass for normal
            x = sample["image_bg"]
            x = x.unsqueeze(0).to(self.device)
            outputs_normal = self.module(x, mode="normal")
            outputs_forces["normal"] = outputs_normal["normal"]

            # forward pass for shear
            x = sample["image"]
            x = x.unsqueeze(0).to(self.device)
            img_fg = sample["image"][0:3].permute(1, 2, 0).cpu().numpy()
            img_fg = cv2.GaussianBlur(img_fg, (5, 5), 0)
            self.img_buffer.append(img_fg)

            outputs_shear = self.module(x, mode="shear")
            outputs_forces["shear"] = outputs_shear["shear"]

            # post-process normal and shear
            normal_unmask = outputs_forces["normal"]
            normal_print = outputs_forces["normal"]
            normal_unmask = self._normalize_image(normal_unmask)

            if normal_unmask.mean() > 0.4:
                normal_unmask = 1.0 - normal_unmask

            mask = self._normal2mask(
                normal_unmask, self.bg_template, border, ratio, clip
            )

            th = self.th_no_contact
            avg_img = np.mean(np.array(self.img_buffer), axis=0)
            # print(f'avg img std = {avg_img.std()} | th_std = {std_img_no_contact}')

            if avg_img.std() <= th:
                mask = torch.zeros_like(mask)

            normal = (normal_unmask * mask).cpu().numpy().squeeze()

            dilate = cv2.dilate(normal, np.ones((5, 5), np.uint8), iterations=3)
            normal = cv2.erode(dilate, np.ones((5, 5), np.uint8), iterations=2)
            normal = cv2.GaussianBlur(normal, (15, 15), 0)

            shear = (
                outputs_forces["shear"]
                .cpu()
                .detach()
                .squeeze()
                .permute(1, 2, 0)
                .numpy()
            )

            if not init_done:
                self._init_shear(shear, normal)
                init_done = True
                print("Calibration completed.")

            im_shear = self.update_shear(shear, normal)

            img_size = (240 * 3, 320 * 3)
            current_tactile_image = cv2.cvtColor(
                current_tactile_image, cv2.COLOR_BGR2RGB
            )
            current_tactile_image = (
                cv2.resize(current_tactile_image, img_size)
            ).astype(np.uint8)
            im_normal = cv2.resize(normal, img_size)
            im_shear = cv2.resize(im_shear[:, :, 0:3], img_size)
            im_shear = cv2.cvtColor(im_shear, cv2.COLOR_BGR2RGB)

            # apply colormap on cv2 image
            im_normal = cv2.applyColorMap(
                (im_normal * 255).astype(np.uint8), cv2.COLORMAP_INFERNO
            )

            # add a white border around current tactile image, im_normal and im_shear
            b = 10
            current_tactile_image = cv2.copyMakeBorder(
                current_tactile_image,
                b,
                b,
                b,
                b,
                cv2.BORDER_CONSTANT,
                value=[255, 255, 255],
            )
            im_normal = cv2.copyMakeBorder(
                im_normal, b, b, b, b, cv2.BORDER_CONSTANT, value=[255, 255, 255]
            )
            im_shear = cv2.copyMakeBorder(
                im_shear, b, b, b, b, cv2.BORDER_CONSTANT, value=[255, 255, 255]
            )

            font = cv2.FONT_HERSHEY_SIMPLEX
            org = (15, 35)
            pos_top = (int(120 * 2.8), 35)
            pos_down = (int(120 * 2.8), 320 * 3)
            fontScale = 1.0
            color2 = (255, 255, 255)
            thickness = 2
            current_tactile_image = cv2.putText(
                current_tactile_image,
                self.sensor_handler.sensor,
                org,
                font,
                fontScale,
                color2,
                thickness,
                cv2.LINE_AA,
            )

            current_tactile_image = cv2.putText(
                current_tactile_image,
                "Top",
                pos_top,
                font,
                fontScale,
                color2,
                thickness,
                cv2.LINE_AA,
            )
            current_tactile_image = cv2.putText(
                current_tactile_image,
                "Bottom",
                pos_down,
                font,
                fontScale,
                color2,
                thickness,
                cv2.LINE_AA,
            )

            im_normal = cv2.putText(
                im_normal,
                "Normal",
                org,
                font,
                fontScale,
                color2,
                thickness,
                cv2.LINE_AA,
            )
            im_shear = cv2.putText(
                im_shear, "Shear", org, font, fontScale, color2, thickness, cv2.LINE_AA
            )
            im_h = cv2.hconcat([current_tactile_image, im_normal, im_shear])
            print(
                f"Normal max: {normal_print.max():.4f} | Normal mean: {normal_print.mean():.4f} | Shear max: {shear.max():.4f} | Shear mean: {shear.mean():.4f}"
            )
            # normal_print = 0.06
            if (len(buffer) >= 10 and np.mean(buffer[-10:]) < 0.0005) or (
                len(shear_buffer) >= 10 and np.mean(shear_buffer[-10:]) < 0.0002
            ) or (len(shear_buffer_max) >= 10 and np.mean(shear_buffer_max[-10:]) < 0.0005):
                print("controller stabilized")
                print(
                    f"Gripper adjustment: {-adjustment:.4f} | Error: {error_mean:.4f} | Target Normal Max: {target_normal_max:.4f}"
                )
                # prompt for a user input to continue
                user_input = input(
                    "Press Enter to continue or type 'exit' to quit: "
                )
                if user_input.strip().lower() == '':
                    print("Done")
                    return tactile_images
                    # self.robot.call("home_robot")
                    # self.robot.call("set_guiding_mode", True)
            else:
                # minimal contact thresholds should be later extracted from signal from image to touch modelt
                print(self.force_controller_parameters)
                target_normal_max = self.force_controller_parameters.get(
                    "target_normal_max", 0.07
                )
                target_shear_max = self.force_controller_parameters.get(
                    "target_shear_max", 0.32
                )
                target_shear_mean = self.force_controller_parameters.get(
                    "target_shear_mean", 0.04
                )
                ##################################################

                shear_error_mean = target_shear_mean - np.mean(shear)
                shear_error_max = target_shear_max - np.max(shear)
                kp = self.force_controller_parameters.get("kp_slow",0.005)  # Proportional gain
                kd = self.force_controller_parameters.get("kd",0.002)  # Derivative gain
                error_mean = target_normal_max - np.mean(
                    normal_print.cpu().detach().numpy()
                )
                error_max = target_normal_max - np.max(
                    normal_print.cpu().detach().numpy()
                )
                # error = target_normal_max - normal_print
                delta_error = error_mean - prev_error
                delta_error_max = error_max - prev_error
                delta_shear_mean = shear_error_mean - prev_shear_error
                delta_shear_max = shear_error_max - prev_shear_error_max
                prev_error = error_mean
                prev_shear_error = shear_error_mean
                prev_shear_error_max = shear_error_max
                # #make kp adaptive based on std of mean and max of predicted normal and shear forces
                if (
                    (error_mean < self.force_controller_parameters.get("error_threshold",0.003) and error_mean >= 0)
                    or (error_max < self.force_controller_parameters.get("error_threshold",0.003) and error_max >= 0)
                    or delta_error <= -self.force_controller_parameters.get("error_threshold",0.003)
                    or (shear_error_mean < self.force_controller_parameters.get("error_threshold",0.003) and shear_error_mean >= 0)
                    or (shear_error_max < self.force_controller_parameters.get("error_threshold",0.003) and shear_error_max >= 0)
                    or delta_shear_mean <= -self.force_controller_parameters.get("delta_error_threshold",0.005)
                    or delta_shear_max <= -self.force_controller_parameters.get("delta_shear_error_threshold",0.03)
                ):
                    kp = self.force_controller_parameters.get("kp_slow",0.005)
                    print("contact detected, slow mode")
                    buffer.append(error_mean)
                    shear_buffer.append(shear_error_mean)
                    shear_buffer_max.append(shear_error_max)
                elif error_mean >= self.force_controller_parameters.get("error_threshold",0.003):
                    kp = self.force_controller_parameters.get("kp_fast",8)
                adjustment = kp * error_mean + kd * delta_error

                self._adjust_gripper(-adjustment)
                time.sleep(1)
                print(
                    f"Gripper adjustment: {-adjustment:.4f} | Error mean: {error_mean:.4f} | Target Normal Max: {target_normal_max:.4f}"
                )
                # display delta error and delta_error_max and error max
                print(
                    f"Delta Error Mean: {delta_error:.4f} | Delta Error Max: {delta_error_max:.4f} | Error Max: {error_max:.4f}"
                )
        # breakpoint()
        # cv2.imshow('sparsh', im_h)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

        # cv2.destroyAllWindows()
        # plt.close(self.fig)
