import time
import json

try:
    import warp as wp
    WARP_AVAILABLE = True
except ImportError:
    WARP_AVAILABLE = False

class SimulationProfiler:
    def __init__(self, device: str, warmup_steps: int = 5):
        self.device = str(device)
        self.is_gpu = "cuda" in self.device.lower()
        self.warmup_steps = warmup_steps if self.is_gpu else 0
        
        self.current_step = 0
        # Initialize start time immediately for the CPU (which has 0 warmup steps)
        self.run_start_time = time.perf_counter() 
        self.t_step_start = 0.0
        self.t_compute_start = 0.0
        
        self.compute_ms = 0.0
        self.step_time_ms = 0.0

    def sync(self):
        """Force the CPU to wait for the GPU to finish all queued work."""
        if self.is_gpu and WARP_AVAILABLE:
            wp.synchronize_device(self.device)

    def start_run(self):
        self.sync()
        self.run_start_time = time.perf_counter()

    def start_step(self):
        self.sync()
        self.t_step_start = time.perf_counter()

    def start_compute(self):
        self.sync()
        self.t_compute_start = time.perf_counter()

    def end_compute(self):
        self.sync()
        self.compute_ms = (time.perf_counter() - self.t_compute_start) * 1000.0

    def end_step(self, metrics_dict: dict, logger):
        self.sync()
        self.step_time_ms = (time.perf_counter() - self.t_step_start) * 1000.0
        self.current_step += 1

        # Skip logging during the JIT warmup phase
        if self.current_step <= self.warmup_steps:
            if self.current_step == self.warmup_steps:
                self.start_run()
            return

        total_time_ms = (time.perf_counter() - self.run_start_time) * 1000.0

        metrics_dict["compute_ms"] = self.compute_ms
        metrics_dict["step_time_ms"] = self.step_time_ms
        metrics_dict["total_time_ms"] = total_time_ms
        metrics_dict["device"] = self.device
        
        logger.log_metrics(metrics_dict)