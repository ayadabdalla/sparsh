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

from typing import Optional, List, Union, Dict
from tactile_ssl import algorithm

import torch
import torch.nn.functional as F
from tqdm import tqdm
from scipy import stats
from PIL import Image
from collections import deque
from omegaconf import OmegaConf

from .test_task import TestTaskSL

from tactile_ssl.data.vision_based_interactive import MultiDemoForceFieldData
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


class MultiDemoForceField(TestTaskSL):
    def __init__(
        self,
        digit_serials: Union[str, List[str]],
        gelsight_device_ids: Union[int, List[int]] = None,
        device=None,
        module: algorithm.Module = None,
        digit_sensors: Optional[List] = None,
        robot=None,
        force_controller_parameters: Optional[dict] = None,
        sensor_combination_method: str = "min",  # "average", "max", "concat", "separate"
        sensor_weights: Optional[List[float]] = None,
    ):
        super().__init__(
            device=device,
            module=module,
        )
        self.robot = robot
        self.digit_sensors = digit_sensors or []
        
        # Convert single inputs to lists
        if isinstance(digit_serials, str):
            digit_serials = [digit_serials]
        if isinstance(gelsight_device_ids, int):
            gelsight_device_ids = [gelsight_device_ids]
            
        self.digit_serials = digit_serials
        self.gelsight_device_ids = gelsight_device_ids or []
        self.num_sensors = len(digit_serials) if digit_serials else len(gelsight_device_ids)
        
        self.force_controller_parameters = force_controller_parameters
        self.sensor_combination_method = sensor_combination_method
        self.sensor_weights = sensor_weights or [1.0] * self.num_sensors
        
        # print("initializing RPC client...")
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
        
        # Initialize multi-sensor handler
        self.sensor_handler = MultiDemoForceFieldData(
            config=self.config.data.dataset.config,
            digit_serials=self.digit_serials,
            gelsight_device_ids=self.gelsight_device_ids,
            digit_sensors=self.digit_sensors,
        )
        
        # Initialize image buffers for each sensor
        self.img_buffers = [deque(maxlen=5) for _ in range(self.num_sensors)]
        self._set_bg_templates()

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

    def _set_bg_templates(self):
        """Set background templates for all sensors."""
        self.bg_templates = []
        
        for sensor_idx in range(self.num_sensors):
            bg = self.sensor_handler.backgrounds[sensor_idx]
            bg = compute_diff(bg, bg)
            bg = Image.fromarray(bg)
            bg = self.sensor_handler.transform_resize(bg).unsqueeze(0).to(self.device)
            bg = torch.cat([bg, bg], dim=1)
            outputs_forces = self.module(bg)
            bg_template = self._normalize_image(outputs_forces["normal"]).squeeze()
            self.bg_templates.append(bg_template)

    def _combine_sensor_outputs(self, sensor_outputs: Dict, output_type: str):
        """
        Combine outputs from multiple sensors.
        
        Args:
            sensor_outputs: Dictionary with sensor outputs
            output_type: "normal" or "shear"
        """
        values = []
        for sensor_idx in range(self.num_sensors):
            sensor_key = f"sensor_{sensor_idx}"
            if sensor_key in sensor_outputs:
                values.append(sensor_outputs[sensor_key][output_type])
        
        if not values:
            return None
        
        if self.sensor_combination_method == "average":
            # Weighted average
            weighted_sum = sum(w * v for w, v in zip(self.sensor_weights, values))
            total_weight = sum(self.sensor_weights[:len(values)])
            return weighted_sum / total_weight
        elif self.sensor_combination_method == "max":
            # Element-wise maximum
            return torch.max(torch.stack(values), dim=0)[0]
        elif self.sensor_combination_method == "concat":
            # Concatenate along channel dimension
            return torch.cat(values, dim=0)
        elif self.sensor_combination_method == "separate":
            # Return all values separately
            return values
        elif self.sensor_combination_method == "sum":
            # Element-wise sum
            return torch.sum(torch.stack(values), dim=0)
        elif self.sensor_combination_method == "min":
            # Element-wise minimum
            return torch.min(torch.stack(values), dim=0)[0]
        else:
            raise ValueError(f"Unknown combination method: {self.sensor_combination_method}")

    def _init_shear_multi(self, shears, normals, margin=0, spacing=12):
        """Initialize shear visualization for multiple sensors."""
        num_sensors = len(shears) if isinstance(shears, list) else 1
        
        if num_sensors == 1:
            # Single sensor case
            shear = shears[0] if isinstance(shears, list) else shears
            normal = normals[0] if isinstance(normals, list) else normals
            self._init_shear_single(shear, normal, margin, spacing)
        else:
            # Multiple sensors case
            self.fig, self.axes = plt.subplots(1, num_sensors, figsize=(8 * num_sensors, 8))
            self.fig.patch.set_facecolor("black")
            
            if num_sensors == 1:
                self.axes = [self.axes]
            
            self.ax_shears = []
            
            for i, (shear, normal) in enumerate(zip(shears, normals)):
                ax = self.axes[i]
                ax_shear = self._setup_single_shear_plot(shear, normal, ax, margin, spacing)
                self.ax_shears.append(ax_shear)
                ax.set_title(f"Sensor {i}", color='white', fontsize=12)

    def _init_shear_single(self, shear, normal, margin=0, spacing=12):
        """Initialize shear visualization for single sensor (backward compatibility)."""
        self.fig, self.ax = plt.subplots()
        self.fig.patch.set_facecolor("black")
        self.ax_shear = self._setup_single_shear_plot(shear, normal, self.ax, margin, spacing)

    def _setup_single_shear_plot(self, shear, normal, ax, margin=0, spacing=12):
        """Set up a single shear plot."""
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
        ax_shear = ax.quiver(y, x, uu, -vv, m, **kwargs)
        ax.set_ylim(sorted(ax.get_ylim(), reverse=True))
        ax.set_facecolor("black")
        ax.set_xticks([])
        ax.set_yticks([])

        # remove white border
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)

        cbar = self.fig.colorbar(ax_shear, ax=ax, orientation="vertical")
        cbar.set_label(
            "Normalized normal force", labelpad=0.05, fontsize=10, color="white"
        )
        cbar.set_ticks([0, 1])
        cbar.ax.tick_params(labelcolor="white", labelsize=10)
        
        return ax_shear

    def update_shear_multi(self, shears, normals, margin=0, spacing=12):
        """Update shear visualization for multiple sensors."""
        if isinstance(shears, list) and len(shears) > 1:
            # Multiple sensors
            for i, (shear, normal) in enumerate(zip(shears, normals)):
                self._update_single_shear(shear, normal, self.ax_shears[i], margin, spacing)
        else:
            # Single sensor (backward compatibility)
            shear = shears[0] if isinstance(shears, list) else shears
            normal = normals[0] if isinstance(normals, list) else normals
            self._update_single_shear(shear, normal, self.ax_shear, margin, spacing)

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

    def _update_single_shear(self, shear, normal, ax_shear, margin=0, spacing=12):
        """Update a single shear plot."""
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
        v = v / (rad_max + epsilon)

        uu = u.copy()
        vv = v.copy()
        r = np.sqrt(u**2 + v**2)
        idx_clip = np.where(r < 0.01)
        uu[idx_clip] = 0.0
        vv[idx_clip] = 0.0

        uu = uu / (np.abs(uu).max() + epsilon)
        vv = vv / (np.abs(vv).max() + epsilon)

        ax_shear.set_UVC(uu, -vv, m)
        ax_shear.set_clim(0.0, 1.0)

    def run_model(self):
        # self.robot.call("set_gripper_width", 1)
        # self.robot.call("set_guiding_mode", True)
        border, ratio, clip = 15, 1.0, 50
        prev_error = 0.0
        prev_shear_error = 0.0
        prev_shear_error_max = 0.0
        buffer = []
        shear_buffer = []
        shear_buffer_max = []
        tactile_images = []
        init_done = False

        print(f"Multi-Sparsh demo starting with {self.num_sensors} sensors. Press 'q' to exit the demo.")
        print(f"Sensor combination method: {self.sensor_combination_method}")

        print("Starting range calibration ...")
        print("Please do not touch the sensors.")
        
        # Set digit sensors if provided
        if self.digit_sensors:
            for i, sensor in enumerate(self.digit_sensors):
                if i < len(self.sensor_handler.touch_sensors):
                    self.sensor_handler.touch_sensors[i] = sensor

        # Calibration phase
        for _ in range(5):
            all_inputs = self.sensor_handler.get_synchronized_inputs()
            
            for sensor_idx in range(self.num_sensors):
                sensor_key = f"sensor_{sensor_idx}"
                if sensor_key in all_inputs:
                    sample = all_inputs[sensor_key]
                    img_fg = sample["image"][0:3].permute(1, 2, 0).cpu().numpy()
                    img_fg = cv2.GaussianBlur(img_fg, (5, 5), 0)
                    self.img_buffers[sensor_idx].append(img_fg)
            
            time.sleep(0.1)

        # Compute average images and std for each sensor
        avg_imgs_no_contact = []
        std_imgs_no_contact = []
        
        for sensor_idx in range(self.num_sensors):
            avg_img = np.mean(np.array(self.img_buffers[sensor_idx]), axis=0)
            std_img = np.std(avg_img) * 1.7
            avg_imgs_no_contact.append(avg_img)
            std_imgs_no_contact.append(std_img)

        counter = 0

        while True:
            # Get synchronized inputs from all sensors
            all_inputs = self.sensor_handler.get_synchronized_inputs()
            
            # Collect current tactile images
            current_tactile_images = []
            for sensor_idx in range(self.num_sensors):
                sensor_key = f"sensor_{sensor_idx}"
                if sensor_key in all_inputs:
                    current_tactile_images.append(all_inputs[sensor_key]["current_image_color"])
            
            tactile_images.append(current_tactile_images)

            # Process each sensor
            sensor_outputs = {}
            sensor_normals = []
            sensor_shears = []
            sensor_masks = []

            for sensor_idx in range(self.num_sensors):
                sensor_key = f"sensor_{sensor_idx}"
                if sensor_key not in all_inputs:
                    continue
                    
                sample = all_inputs[sensor_key]
                
                # Forward pass for normal
                x = sample["image_bg"]
                x = x.unsqueeze(0).to(self.device)
                outputs_normal = self.module(x, mode="normal")
                
                # Forward pass for shear
                x = sample["image"]
                x = x.unsqueeze(0).to(self.device)
                img_fg = sample["image"][0:3].permute(1, 2, 0).cpu().numpy()
                img_fg = cv2.GaussianBlur(img_fg, (5, 5), 0)
                self.img_buffers[sensor_idx].append(img_fg)

                outputs_shear = self.module(x, mode="shear")
                
                sensor_outputs[sensor_key] = {
                    "normal": outputs_normal["normal"],
                    "shear": outputs_shear["shear"]
                }

                # Post-process normal and shear for this sensor
                normal_unmask = outputs_normal["normal"]
                normal_print = outputs_normal["normal"]
                normal_unmask = self._normalize_image(normal_unmask)

                if normal_unmask.mean() > 0.4:
                    normal_unmask = 1.0 - normal_unmask

                mask = self._normal2mask(
                    normal_unmask, self.bg_templates[sensor_idx], border, ratio, clip
                )

                th = self.th_no_contact
                avg_img = np.mean(np.array(self.img_buffers[sensor_idx]), axis=0)

                if avg_img.std() <= th:
                    mask = torch.zeros_like(mask)

                normal = (normal_unmask * mask).cpu().numpy().squeeze()

                dilate = cv2.dilate(normal, np.ones((5, 5), np.uint8), iterations=3)
                normal = cv2.erode(dilate, np.ones((5, 5), np.uint8), iterations=2)
                normal = cv2.GaussianBlur(normal, (15, 15), 0)

                shear = (
                    outputs_shear["shear"]
                    .cpu()
                    .detach()
                    .squeeze()
                    .permute(1, 2, 0)
                    .numpy()
                )

                sensor_normals.append(normal)
                sensor_shears.append(shear)
                sensor_masks.append(mask)

            if not sensor_normals:
                continue

            # Initialize visualization on first frame
            if not init_done:
                self._init_shear_multi(sensor_shears, sensor_normals)
                init_done = True
                print("Calibration completed.")

            # Update visualization
            im_shear = self.update_shear_multi(sensor_shears, sensor_normals)

            # Create combined visualization
            img_size = (240 * 3, 320 * 3)
            combined_images = []

            for i, current_tactile_image in enumerate(current_tactile_images):
                current_tactile_image = cv2.cvtColor(
                    current_tactile_image, cv2.COLOR_BGR2RGB
                )
                current_tactile_image = (
                    cv2.resize(current_tactile_image, img_size)
                ).astype(np.uint8)
                
                im_normal = cv2.resize(sensor_normals[i], img_size)
                im_normal = cv2.applyColorMap(
                    (im_normal * 255).astype(np.uint8), cv2.COLORMAP_INFERNO
                )

                # Add borders and labels
                b = 10
                current_tactile_image = cv2.copyMakeBorder(
                    current_tactile_image,
                    b, b, b, b,
                    cv2.BORDER_CONSTANT,
                    value=[255, 255, 255],
                )
                im_normal = cv2.copyMakeBorder(
                    im_normal, b, b, b, b, cv2.BORDER_CONSTANT, value=[255, 255, 255]
                )

                font = cv2.FONT_HERSHEY_SIMPLEX
                org = (15, 35)
                fontScale = 1.0
                color2 = (255, 255, 255)
                thickness = 2

                current_tactile_image = cv2.putText(
                    current_tactile_image,
                    f"Sensor {i}",
                    org,
                    font,
                    fontScale,
                    color2,
                    thickness,
                    cv2.LINE_AA,
                )

                im_normal = cv2.putText(
                    im_normal,
                    f"Normal {i}",
                    org,
                    font,
                    fontScale,
                    color2,
                    thickness,
                    cv2.LINE_AA,
                )

                combined_images.extend([current_tactile_image, im_normal])

            # Add shear visualization
            im_shear_resized = cv2.resize(im_shear[:, :, 0:3], img_size)
            im_shear_resized = cv2.cvtColor(im_shear_resized, cv2.COLOR_BGR2RGB)
            im_shear_resized = cv2.copyMakeBorder(
                im_shear_resized, b, b, b, b, cv2.BORDER_CONSTANT, value=[255, 255, 255]
            )
            im_shear_resized = cv2.putText(
                im_shear_resized,
                "Combined Shear",
                org,
                font,
                fontScale,
                color2,
                thickness,
                cv2.LINE_AA,
            )
            combined_images.append(im_shear_resized)

            # Combine all sensor data for control
            if self.sensor_combination_method == "separate":
                # Use first sensor for control (or implement more sophisticated logic)
                combined_normal = sensor_outputs[f"sensor_0"]["normal"]
                combined_shear_tensor = sensor_outputs[f"sensor_0"]["shear"]
                combined_shear = sensor_shears[0]
            else:
                combined_normal = self._combine_sensor_outputs(sensor_outputs, "normal")
                combined_shear_tensor = self._combine_sensor_outputs(sensor_outputs, "shear")
                
                # For numpy arrays, compute weighted average
                if self.sensor_combination_method == "average":
                    combined_shear = np.average(sensor_shears, axis=0, weights=self.sensor_weights[:len(sensor_shears)])
                elif self.sensor_combination_method == "max":
                    combined_shear = np.max(sensor_shears, axis=0)
                elif self.sensor_combination_method == "sum":
                    combined_shear = np.sum(sensor_shears, axis=0)
                else:
                    combined_shear = sensor_shears[0]  # fallback

            print(
                f"Normal max: {combined_normal.max():.4f} | Normal mean: {combined_normal.mean():.4f} | Shear max: {combined_shear.max():.4f} | Shear mean: {combined_shear.mean():.4f}"
            )
            

            if (len(buffer) >= 15 and np.mean(buffer[-15:]) < 0.0005) or (
                len(shear_buffer) >= 15 and np.mean(shear_buffer[-15:]) < 0.0002
            ) or (len(shear_buffer_max) >= 15 and np.mean(shear_buffer_max[-15:]) < 0.0005):
                print("controller stabilized")
                print(
                    f"Gripper adjustment: {-adjustment:.4f} | Error: {error_mean:.4f} | Target Normal Max: {target_normal_max:.4f}"
                )
                user_input = input(
                    "Press Enter to continue or type 'exit' to quit: "
                )
                if user_input.strip().lower() == '':
                    print("Done")
                    return tactile_images
            else:
                print(self.force_controller_parameters)
                target_normal_max = self.force_controller_parameters.get(
                    "target_normal_max", 0.075
                )
                target_shear_max = self.force_controller_parameters.get(
                    "target_shear_max", 0.33
                )
                target_shear_mean = self.force_controller_parameters.get(
                    "target_shear_mean", 0.045
                )

                shear_error_mean = target_shear_mean - np.mean(combined_shear)
                shear_error_max = target_shear_max - np.max(combined_shear)
                kp = self.force_controller_parameters.get("kp_slow", 0.005)
                kd = self.force_controller_parameters.get("kd", 0.002)
                error_mean = target_normal_max - np.mean(
                    combined_normal.cpu().detach().numpy()
                )
                error_max = target_normal_max - np.max(
                    combined_normal.cpu().detach().numpy()
                )
                
                delta_error = error_mean - prev_error
                delta_error_max = error_max - prev_error
                delta_shear_mean = shear_error_mean - prev_shear_error
                delta_shear_max = shear_error_max - prev_shear_error_max
                prev_error = error_mean
                prev_shear_error = shear_error_mean
                prev_shear_error_max = shear_error_max

                if (
                    (error_mean < self.force_controller_parameters.get("error_threshold", 0.003) and error_mean >= 0)
                    or (error_max < self.force_controller_parameters.get("error_threshold", 0.003) and error_max >= 0)
                    or delta_error <= -self.force_controller_parameters.get("error_threshold", 0.003)
                    or (shear_error_mean < self.force_controller_parameters.get("error_threshold", 0.003) and shear_error_mean >= 0)
                    or (shear_error_max < self.force_controller_parameters.get("error_threshold", 0.003) and shear_error_max >= 0)
                    or delta_shear_mean <= -self.force_controller_parameters.get("delta_error_threshold", 0.005)
                    or delta_shear_max <= -self.force_controller_parameters.get("delta_shear_error_threshold", 0.03)
                ):
                    print("the reason for adjustment to slow mode is:")
                    print(f"error_mean: {error_mean:.4f} | error_max: {error_max:.4f}")
                    print(f"shear_error_mean: {shear_error_mean:.4f} | shear_error_max: {shear_error_max:.4f}")
                    print(f"delta_error: {delta_error:.4f}")
                    print(f"delta_shear_mean: {delta_shear_mean:.4f} | delta_shear_max: {delta_shear_max:.4f}")
                    
                    kp = self.force_controller_parameters.get("kp_slow", 0.005)
                    print("contact detected, slow mode")
                    buffer.append(error_mean)
                    shear_buffer.append(shear_error_mean)
                    shear_buffer_max.append(shear_error_max)
                elif error_mean >= self.force_controller_parameters.get("error_threshold", 0.003):
                    if len(buffer) < 10:
                        kp = self.force_controller_parameters.get("kp_fast", 8)
                    else:
                        kp = self.force_controller_parameters.get("kp_slow", 0.005)
                        print("entered slow mode many times, will not change to fast mode")
                    
                adjustment = kp * error_mean + kd * delta_error

                self._adjust_gripper(-adjustment)
                time.sleep(1)
                print(
                    f"Gripper adjustment: {-adjustment:.4f} | Error mean: {error_mean:.4f} | Target Normal Max: {target_normal_max:.4f}"
                )
                print(
                    f"Delta Error Mean: {delta_error:.4f} | Delta Error Max: {delta_error_max:.4f} | Error Max: {error_max:.4f}"
                )