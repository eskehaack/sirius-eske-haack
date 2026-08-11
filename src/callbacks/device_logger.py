import threading
import time
from typing import Optional

import amdsmi
import lightning as L


class AMDGPUMonitor(L.Callback):
    """
    Monitor AMD GPU utilization using AMD SMI and log metrics to the
    PyTorch Lightning logger (e.g. TensorBoard).

    Metrics:
        gpu/gfx_utilization
        gpu/memory_utilization
        gpu/vram_used_mb
        gpu/vram_total_mb
    """

    def __init__(
        self,
        interval: float = 1.0,
        device_index: int = 0,
    ):
        super().__init__()

        self.interval = interval
        self.device_index = device_index

        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        self._lock = threading.Lock()
        self._latest_metrics = {}

        self._device = None
        self._initialized = False

    def setup(self, trainer, pl_module, stage=None):
        """Initialize AMD SMI and select the GPU."""

        if self._initialized:
            return

        amdsmi.amdsmi_init()

        devices = amdsmi.amdsmi_get_processor_handles()

        if not devices:
            raise RuntimeError("No AMD GPUs found by AMD SMI.")

        if self.device_index >= len(devices):
            raise ValueError(
                f"GPU index {self.device_index} does not exist. "
                f"Found {len(devices)} GPU(s)."
            )

        self._device = devices[self.device_index]
        self._initialized = True

    def _sample_gpu(self):
        """Read one sample from AMD SMI."""

        if self._device is None:
            return

        try:
            activity = amdsmi.amdsmi_get_gpu_activity(self._device)

            metrics = {
                "gpu/gfx_utilization": float(
                    activity.get("gfx_activity", 0.0)
                ),
                "gpu/memory_utilization": float(
                    activity.get("umc_activity", 0.0)
                ),
            }

            # VRAM usage
            try:
                memory = amdsmi.amdsmi_get_gpu_vram_usage(
                    self._device
                )

                metrics["gpu/vram_used_gb"] = (
                    memory["vram_used"] / 1024
                )

                metrics["gpu/vram_total_gb"] = (
                    memory["vram_total"] / 1024
                )

            except Exception:
                pass

            try:
                power_info = amdsmi.amdsmi_get_power_info(self._device)
                metrics["gpu/average_socket_power"] = power_info["socket_power"]

            except Exception:
                pass

            with self._lock:
                self._latest_metrics = metrics

        except Exception:
            # Don't let a monitoring failure kill training.
            pass

    def _monitor_loop(self):
        """Background monitoring loop."""

        while not self._stop_event.is_set():
            self._sample_gpu()
            self._stop_event.wait(self.interval)

    def on_train_start(self, trainer, pl_module):
        """Start the background GPU monitor."""

        self._stop_event.clear()

        self._thread = threading.Thread(
            target=self._monitor_loop,
            name="amd-gpu-monitor",
            daemon=True,
        )

        self._thread.start()

    def on_train_batch_end(
        self,
        trainer,
        pl_module,
        outputs,
        batch,
        batch_idx,
    ):
        """Log the most recent GPU sample."""

        with self._lock:
            metrics = dict(self._latest_metrics)

        if not metrics:
            return

        # Use the Lightning global step so the values line up
        # with the rest of the TensorBoard metrics.
        for name, value in metrics.items():
            pl_module.log(
                name,
                value,
                on_step=True,
                on_epoch=False,
                logger=True,
                prog_bar=False,
                rank_zero_only=True,
            )

    def on_train_end(self, trainer, pl_module):
        """Stop the monitoring thread."""

        self._stop_event.set()

        if self._thread is not None:
            self._thread.join(timeout=self.interval + 1.0)

        if self._initialized:
            try:
                amdsmi.amdsmi_shut_down()
            except Exception:
                pass

            self._initialized = False