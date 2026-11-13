"""Offline workspace runner for B601; no hardware or simulator is created."""

from diffusion_policy.env_runner.base_image_runner import BaseImageRunner


class OfflineImageRunner(BaseImageRunner):
    def run(self, policy):
        return {"offline/rollouts": 0}
