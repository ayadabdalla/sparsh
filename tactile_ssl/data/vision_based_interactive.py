# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC-BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

from collections import deque
import time
import cv2
from omegaconf import DictConfig
import numpy as np
from PIL import Image
from typing import List, Dict, Optional, Union

import torch
import matplotlib.pyplot as plt

from tactile_ssl.data.digit.utils import (
    load_sample,
    get_resize_transform,
)

from digit_interface.digit import Digit


class MultiDemoForceFieldData:
    def __init__(
        self,
        config: DictConfig,
        digit_serials: Union[str, List[str]],
        gelsight_device_ids: Union[int, List[int]] = None,
        digit_sensors: Optional[List[Digit]] = None,
    ):
        """
        Initialize multi-sensor tactile data collection.
        
        Args:
            config: Configuration object
            digit_serials: Single serial number or list of DIGIT sensor serial numbers
            gelsight_device_ids: Single device ID or list of GelSight device IDs
            digit_sensors: Optional list of pre-initialized DIGIT sensors
        """
        super().__init__()
        self.config = config
        self.sensor = self.config.sensor
        
        # Convert single inputs to lists for uniform handling
        if isinstance(digit_serials, str):
            digit_serials = [digit_serials]
        if isinstance(gelsight_device_ids, int):
            gelsight_device_ids = [gelsight_device_ids]
            
        self.digit_serials = digit_serials
        self.gelsight_device_ids = gelsight_device_ids or []
        self.num_sensors = len(digit_serials) if digit_serials else len(gelsight_device_ids)
        
        self.enhance_diff_img = True if self.sensor == "gelsight_mini" else False

        self.remove_bg = (
            self.config.remove_bg if hasattr(self.config, "remove_bg") else False
        )
        self.out_format = self.config.out_format  # if output video
        assert self.out_format in [
            "video",
            "concat_ch_img",
            "single_image",
        ], ValueError(
            "out_format should be 'video' or 'concat_ch_img' or 'single_image'"
        )

        frame_stride = self.config.frame_stride
        self.num_frames = (
            1 if self.out_format == "single_image" else self.config.num_frames
        )
        self.frames_concat_idx = np.arange(
            0, self.num_frames * frame_stride, frame_stride
        )

        # load dataset
        self.loader = load_sample

        # transforms
        self.img_sz = self.config.transforms.resize
        self.transform_resize = get_resize_transform(self.img_sz)

        # tactile windows for each sensor. Make FIFO buffers with length 6 using deque
        self.tactile_windows = [deque(maxlen=6) for _ in range(self.num_sensors)]

        # connect to sensors
        self.fps = 30.0
        self.touch_sensors = []
        self.backgrounds = []
        
        if self.sensor == "digit":
            self._initialize_digit_sensors(digit_sensors)
        elif "gelsight" in self.sensor:
            self._initialize_gelsight_sensors()
        else:
            raise ValueError("Sensor not supported")

    def _initialize_digit_sensors(self, digit_sensors: Optional[List[Digit]] = None):
        """Initialize multiple DIGIT sensors."""
        if digit_sensors is not None:
            self.touch_sensors = digit_sensors
        else:
            self.touch_sensors = []
            for serial in self.digit_serials:
                sensor = self.connect_digit(serial)
                self.touch_sensors.append(sensor)
        
        # Initialize each sensor and get background images
        for i, sensor in enumerate(self.touch_sensors):
            self._init_digit_sensor(i)
            bg = self.get_digit_image(i)
            self.backgrounds.append(bg)

    def _initialize_gelsight_sensors(self):
        """Initialize multiple GelSight sensors."""
        for device_id in self.gelsight_device_ids:
            sensor = self.connect_gelsight(device_id)
            self.touch_sensors.append(sensor)
        
        # Initialize each sensor and get background images
        for i in range(len(self.touch_sensors)):
            self._init_gelsight_sensor(i)
            bg = self.get_gelsight_image(i)
            self.backgrounds.append(bg)

    def connect_gelsight(self, device_id: int):
        """Connect to a single GelSight sensor."""
        return cv2.VideoCapture(device_id)

    def _init_gelsight_sensor(self, sensor_idx: int):
        """Initialize a specific GelSight sensor."""
        for i in range(100):
            _ = self.get_gelsight_image(sensor_idx)
            time.sleep(1 / self.fps)

    def connect_digit(self, serial: str):
        """Connect to a single DIGIT sensor."""
        digit_sensor = Digit(serial, f"Digit_{serial}")
        digit_sensor.connect()
        digit_sensor.set_intensity(Digit.LIGHTING_MAX)
        # Change DIGIT resolution to QVGA
        qvga_res = Digit.STREAMS["QVGA"]
        digit_sensor.set_resolution(qvga_res)
        fps_30 = Digit.STREAMS["QVGA"]["fps"]["30fps"]
        digit_sensor.set_fps(fps_30)
        # Print device info
        print(f"Connected to DIGIT sensor: {digit_sensor.info()}")
        return digit_sensor

    def _init_digit_sensor(self, sensor_idx: int):
        """Initialize a specific DIGIT sensor."""
        for i in range(100):
            _ = self.get_digit_image(sensor_idx)
            time.sleep(1 / self.fps)

    def _process_image(self, tactile_image):
        """Process a single tactile image."""
        tactile_image = cv2.cvtColor(tactile_image, cv2.COLOR_BGR2RGB)
        h, w, _ = tactile_image.shape
        if h < w:
            tactile_image = cv2.rotate(tactile_image, cv2.ROTATE_90_CLOCKWISE)

        h, w, _ = tactile_image.shape
        r = 4/3 # default aspect ratio
        if h/w != r:
            h2, w2 = int(h/r), w
            tactile_image = tactile_image[int((h-h2)/2):int((h+h2)/2), int((w-w2)/2):int((w+w2)/2)]
        return tactile_image

    def get_digit_image(self, sensor_idx: int):
        """Get image from a specific DIGIT sensor."""
        tactile_image = self.touch_sensors[sensor_idx].get_frame_safe()
        tactile_image = cv2.flip(tactile_image, 1)
        tactile_image = self._process_image(tactile_image)
        self.tactile_windows[sensor_idx].append(tactile_image)
        return tactile_image

    def get_gelsight_image(self, sensor_idx: int):
        """Get image from a specific GelSight sensor."""
        ret, tactile_image = self.touch_sensors[sensor_idx].read()
        tactile_image = cv2.flip(tactile_image, 0)
        tactile_image = cv2.resize(tactile_image, (320, 240))
        tactile_image = self._process_image(tactile_image)
        self.tactile_windows[sensor_idx].append(tactile_image)
        return tactile_image

    def get_all_images(self):
        """Get current images from all sensors."""
        images = []
        if self.sensor == "digit":
            for i in range(self.num_sensors):
                img = self.get_digit_image(i)
                images.append(img)
        elif "gelsight" in self.sensor:
            for i in range(self.num_sensors):
                img = self.get_gelsight_image(i)
                images.append(img)
        else:
            raise ValueError("Sensor not supported")
        return images

    def _plot_tactile_clip(self, clips: List[torch.Tensor], sensor_names: List[str] = None):
        """Plot tactile clips from multiple sensors."""
        if sensor_names is None:
            sensor_names = [f"Sensor_{i}" for i in range(len(clips))]
        
        fig, axs = plt.subplots(len(clips), self.num_frames, figsize=(20, 5 * len(clips)))
        if len(clips) == 1:
            axs = axs.reshape(1, -1)
        
        for sensor_idx, clip in enumerate(clips):
            for frame_idx in range(self.num_frames):
                axs[sensor_idx, frame_idx].imshow(clip[frame_idx].permute(1, 2, 0))
                axs[sensor_idx, frame_idx].axis("off")
                if frame_idx == 0:
                    axs[sensor_idx, frame_idx].set_ylabel(sensor_names[sensor_idx], rotation=90, size='large')
        
        plt.tight_layout()
        plt.savefig("multi_tactile_clip.png")
        plt.close()

    def get_model_inputs(self, sensor_indices: Optional[List[int]] = None):
        """
        Get model inputs from specified sensors or all sensors.
        
        Args:
            sensor_indices: List of sensor indices to use. If None, uses all sensors.
            
        Returns:
            Dictionary with inputs for each sensor
        """
        if sensor_indices is None:
            sensor_indices = list(range(self.num_sensors))
        
        # Get current images from all specified sensors
        current_images = []
        for idx in sensor_indices:
            if self.sensor == "digit":
                img = self.get_digit_image(idx)
            elif "gelsight" in self.sensor:
                img = self.get_gelsight_image(idx)
            else:
                raise ValueError("Sensor not supported")
            current_images.append(img)
        
        # Get tactile inputs for each sensor
        all_inputs = {}
        for i, sensor_idx in enumerate(sensor_indices):
            images, images_bg = self._get_tactile_inputs(sensor_idx, add_bg=True)
            
            sensor_inputs = {
                "image": images,
                "image_bg": images_bg,
                "current_image_color": current_images[i]
            }
            all_inputs[f"sensor_{sensor_idx}"] = sensor_inputs
        
        return all_inputs

    def get_synchronized_inputs(self):
        """
        Get synchronized inputs from all sensors simultaneously.
        This ensures all sensor readings are taken at approximately the same time.
        """
        # Get all current images simultaneously
        current_images = self.get_all_images()
        
        # Process tactile inputs for all sensors
        all_inputs = {}
        for sensor_idx in range(self.num_sensors):
            images, images_bg = self._get_tactile_inputs(sensor_idx, add_bg=True)
            
            sensor_inputs = {
                "image": images,
                "image_bg": images_bg,
                "current_image_color": current_images[sensor_idx]
            }
            all_inputs[f"sensor_{sensor_idx}"] = sensor_inputs
        
        return all_inputs

    def _get_tactile_inputs(self, sensor_idx: int, add_bg: bool = False):
        """Get tactile inputs for a specific sensor."""
        output, output_bg = None, None
        sample_images = []

        for i in self.frames_concat_idx[::-1]:
            img = self.tactile_windows[sensor_idx][i]
            img = self.loader(img, self.backgrounds[sensor_idx], self.enhance_diff_img)
            image = self.transform_resize(img)
            sample_images.append(image)
        
        output = torch.cat(sample_images, dim=0)

        if add_bg:
            bg = self.loader(self.backgrounds[sensor_idx], self.backgrounds[sensor_idx])
            bg = self.transform_resize(bg)
            output_bg = torch.cat([sample_images[0], bg], dim=0)

        return output, output_bg

    def get_combined_inputs(self, combination_method: str = "concat"):
        """
        Get combined inputs from all sensors.
        
        Args:
            combination_method: How to combine sensor data ("concat", "stack", "separate")
            
        Returns:
            Combined sensor inputs
        """
        all_sensor_inputs = self.get_synchronized_inputs()
        
        if combination_method == "separate":
            return all_sensor_inputs
        
        # Extract data from all sensors
        all_images = []
        all_images_bg = []
        all_current_images = []
        
        for sensor_idx in range(self.num_sensors):
            sensor_key = f"sensor_{sensor_idx}"
            all_images.append(all_sensor_inputs[sensor_key]["image"])
            all_images_bg.append(all_sensor_inputs[sensor_key]["image_bg"])
            all_current_images.append(all_sensor_inputs[sensor_key]["current_image_color"])
        
        if combination_method == "concat":
            # Concatenate along channel dimension
            combined_inputs = {
                "image": torch.cat(all_images, dim=0),
                "image_bg": torch.cat(all_images_bg, dim=0),
                "current_image_color": all_current_images,  # Keep as list
                "num_sensors": self.num_sensors
            }
        elif combination_method == "stack":
            # Stack along new sensor dimension
            combined_inputs = {
                "image": torch.stack(all_images, dim=0),
                "image_bg": torch.stack(all_images_bg, dim=0),
                "current_image_color": all_current_images,
                "num_sensors": self.num_sensors
            }
        else:
            raise ValueError(f"Unknown combination method: {combination_method}")
        
        return combined_inputs

    def disconnect_all_sensors(self):
        """Disconnect all sensors properly."""
        if self.sensor == "digit":
            for sensor in self.touch_sensors:
                sensor.disconnect()
        elif "gelsight" in self.sensor:
            for sensor in self.touch_sensors:
                sensor.release()
        print(f"Disconnected {self.num_sensors} sensors")

    def __del__(self):
        """Cleanup when object is destroyed."""
        try:
            self.disconnect_all_sensors()
        except:
            pass  # Ignore errors during cleanup