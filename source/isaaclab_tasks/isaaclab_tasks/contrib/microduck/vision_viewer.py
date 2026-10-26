# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Task-local head-camera panels that borrow the existing Viser server.

The private visualizer/server bridge is isolated here: no core viewer changes,
second server, streaming background, or truth-derived image overlays are needed.
The clockwise display rotation never modifies images used by the tracker.
"""

import math
import time

import numpy as np
import torch


class HeadCameraPanels:
    """Show both raw head images and pixel-tracker status beside the existing 3D scene."""

    def __init__(self, sim, max_fps: float = 30.0):
        if not math.isfinite(max_fps) or max_fps <= 0:
            raise ValueError("The head-camera panel update rate must be finite and positive.")
        self._sim = sim
        self._interval = 1.0 / min(max_fps, 30.0)
        self._next_update = 0.0
        self._panels = {}
        self._stats = {
            "pushes": 0,
            "image_copies": 0,
            "image_copy_wall_s": 0.0,
            "status_copy_wall_s": 0.0,
            "encode_and_push_wall_s": 0.0,
            "total_wall_s": 0.0,
        }

    def _servers(self):
        """Localize the task's only access to implementation-owned Viser objects."""
        for visualizer in self._sim._visualizers:
            if type(visualizer).__name__ != "ViserVisualizer":
                continue
            viewer = getattr(visualizer, "_viewer", None)
            server = getattr(viewer, "_server", None)
            if server is not None:
                yield server

    def _create(self, server, cameras):
        count = min(cameras[0].num_instances, cameras[1].num_instances, 4)
        with server.gui.add_folder("Head cameras", expand_by_default=True) as folder:
            world = server.gui.add_dropdown("World", options=tuple(str(i) for i in range(count)), initial_value="0")
            server.gui.add_markdown("Raw RGB, displayed 90° clockwise. Tracking uses the original camera pixels.")
            images = [
                server.gui.add_image(
                    np.zeros((camera.cfg.width, camera.cfg.height, 3), dtype=np.uint8),
                    label=f"Duck {name}",
                    format="jpeg",
                    jpeg_quality=80,
                )
                for name, camera in zip(("A", "B"), cameras)
            ]
            status = server.gui.add_markdown("Waiting for camera frames.")
        panel = {"folder": folder, "world": world, "images": images, "status": status}
        self._panels[id(server)] = panel
        return panel

    def update(self, cameras, tracks, images=None) -> bool:
        """Push at most 30 wall-clock frames per second, copying only the selected world.

        In headless training or without connected clients, no image is copied
        to the CPU. The method may be called before a Viser viewer is ready;
        binding and GUI construction happen lazily after a client connects.
        ``images`` borrows the environment's last captured RGB buffers without
        triggering new camera captures from the UI. Legacy callers may omit it.
        """
        now = time.monotonic()
        if now < self._next_update or len(cameras) != 2 or len(tracks) != 2:
            return False
        self._next_update = now + self._interval
        pushed = False
        for server in self._servers():
            if not server.get_clients():
                continue
            panel = self._panels.get(id(server)) or self._create(server, cameras)
            world = int(panel["world"].value)
            lines = [f"World **{world}**"]
            copy_started = time.monotonic()
            values = (
                torch.stack(
                    [
                        torch.stack(
                            (track.visible[world], track.valid[world], track.age[world], track.confidence[world])
                        )
                        for track in tracks
                    ]
                )
                .detach()
                .cpu()
                .tolist()
            )
            self._stats["status_copy_wall_s"] += time.monotonic() - copy_started
            for player, (name, camera, handle) in enumerate(zip(("A", "B"), cameras, panel["images"])):
                image_source = images[player] if images is not None else camera.data.output["rgb"].torch
                copy_started = time.monotonic()
                image = image_source[world, ..., :3].detach().cpu().numpy()
                self._stats["image_copy_wall_s"] += time.monotonic() - copy_started
                push_started = time.monotonic()
                handle.image = np.ascontiguousarray(np.rot90(image, k=-1))
                self._stats["encode_and_push_wall_s"] += time.monotonic() - push_started
                self._stats["image_copies"] += 1
                visible, valid, age, confidence = values[player]
                lines.append(
                    f"Duck {name}: visible **{bool(visible)}**, valid **{bool(valid)}**, "
                    f"age {age:.3f} s, confidence {confidence:.2f}"
                )
            panel["status"].content = "\n\n".join(lines)
            self._stats["pushes"] += 1
            pushed = True
        if pushed:
            self._stats["total_wall_s"] += time.monotonic() - now
        return pushed

    def stats(self) -> dict:
        """Return host-only presentation counters without synchronizing the GPU."""
        return {
            **self._stats,
            "max_fps": 1.0 / self._interval,
            "scope": "Head image panels only; excludes the 3D viewer.",
        }
