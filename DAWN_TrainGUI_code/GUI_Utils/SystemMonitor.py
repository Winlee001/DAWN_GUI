"""
System Monitor Module for CPU and GPU metrics monitoring
"""
import psutil
import time

try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False
    print("Warning: pynvml not available. GPU monitoring will be disabled.")

try:
    from PyQt6.QtCore import QThread, pyqtSignal
    PYQT6_AVAILABLE = True
except ImportError:
    PYQT6_AVAILABLE = False
    print("Warning: PyQt6 not available. MonitorThread will not be available.")


class SystemMetrics:
    """Container for system metrics"""
    def __init__(self):
        self.cpu_percent = 0.0
        self.cpu_memory_percent = 0.0
        self.cpu_memory_used_gb = 0.0
        self.cpu_memory_total_gb = 0.0

        self.gpu_count = 0
        self.gpu_metrics = []

    def to_dict(self):
        return {
            "cpu_percent": self.cpu_percent,
            "cpu_memory_percent": self.cpu_memory_percent,
            "cpu_memory_used_gb": self.cpu_memory_used_gb,
            "cpu_memory_total_gb": self.cpu_memory_total_gb,
            "gpu_count": self.gpu_count,
            "gpu_metrics": self.gpu_metrics,
        }


class SystemMonitor:
    """System monitoring class for CPU and GPU metrics"""

    def __init__(self):
        self.pynvml_initialized = False
        self.gpu_available = False
        self.gpu_count = 0

        if PYNVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self.pynvml_initialized = True
                self.gpu_count = pynvml.nvmlDeviceGetCount()
                self.gpu_available = self.gpu_count > 0
                print(f"System Monitor initialized with {self.gpu_count} GPU(s)")
            except Exception as e:
                print(f"Failed to initialize NVML: {e}")
                self.pynvml_initialized = False
                self.gpu_available = False
                self.gpu_count = 0
        else:
            print("NVML not available, GPU monitoring disabled")
            self.gpu_available = False
            self.gpu_count = 0

    def get_cpu_metrics(self):
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()

        return {
            "cpu_percent": cpu_percent,
            "memory_percent": memory.percent,
            "memory_used_gb": memory.used / (1024 ** 3),
            "memory_total_gb": memory.total / (1024 ** 3),
        }

    def get_gpu_metrics(self, device_id=0):
        if not self.pynvml_initialized or not self.gpu_available:
            return None

        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)

            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
            gpu_util = utilization.gpu
            memory_util = utilization.memory

            memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            memory_used_gb = memory_info.used / (1024 ** 3)
            memory_total_gb = memory_info.total / (1024 ** 3)
            memory_percent = (memory_info.used / memory_info.total) * 100

            try:
                temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            except Exception:
                temperature = 0

            try:
                power_usage = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
            except Exception:
                power_usage = 0

            try:
                device_name = pynvml.nvmlDeviceGetName(handle)
                if isinstance(device_name, bytes):
                    device_name = device_name.decode("utf-8")
            except Exception:
                device_name = f"GPU {device_id}"

            return {
                "device_id": device_id,
                "device_name": device_name,
                "gpu_util": gpu_util,
                "memory_util": memory_util,
                "memory_used_gb": memory_used_gb,
                "memory_total_gb": memory_total_gb,
                "memory_percent": memory_percent,
                "temperature": temperature,
                "power_usage": power_usage,
            }
        except Exception as e:
            print(f"Error getting GPU {device_id} metrics: {e}")
            return None

    def get_all_metrics(self):
        metrics = SystemMetrics()

        cpu_metrics = self.get_cpu_metrics()
        metrics.cpu_percent = cpu_metrics["cpu_percent"]
        metrics.cpu_memory_percent = cpu_metrics["memory_percent"]
        metrics.cpu_memory_used_gb = cpu_metrics["memory_used_gb"]
        metrics.cpu_memory_total_gb = cpu_metrics["memory_total_gb"]

        if self.gpu_available:
            metrics.gpu_count = self.gpu_count
            for i in range(self.gpu_count):
                gpu_metric = self.get_gpu_metrics(i)
                if gpu_metric:
                    metrics.gpu_metrics.append(gpu_metric)

        return metrics

    def shutdown(self):
        if self.pynvml_initialized:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass


if PYQT6_AVAILABLE:
    class MonitorThread(QThread):
        """Thread for continuous system monitoring"""
        metricsUpdated = pyqtSignal(object)

        def __init__(self, parent=None, interval=1.0):
            super().__init__(parent)
            self.interval = interval
            self.monitor = SystemMonitor()
            self.running = False

        def run(self):
            self.running = True
            while self.running:
                try:
                    metrics = self.monitor.get_all_metrics()
                    self.metricsUpdated.emit(metrics)
                    time.sleep(self.interval)
                except Exception as e:
                    print(f"Error in monitoring thread: {e}")
                    time.sleep(self.interval)

        def stop(self):
            self.running = False
            self.wait()
            self.monitor.shutdown()
else:
    class MonitorThread:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyQt6 is required for MonitorThread. Please install PyQt6.")
