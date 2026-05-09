# Copyright (c) OpenMMLab. All rights reserved.
import time

from mmengine.hooks import Hook

from mmdet3d.registry import HOOKS


@HOOKS.register_module()
class EpochSleepHook(Hook):
    """Sleep after each training epoch to let local hardware cool down."""

    def __init__(self, sleep_seconds: int = 180) -> None:
        self.sleep_seconds = int(sleep_seconds)
        if self.sleep_seconds < 0:
            raise ValueError('sleep_seconds must be >= 0.')

    def after_train_epoch(self, runner) -> None:
        if self.sleep_seconds == 0:
            return
        runner.logger.info(
            f'EpochSleepHook: sleep {self.sleep_seconds}s after epoch '
            f'{runner.epoch + 1}.')
        time.sleep(self.sleep_seconds)
