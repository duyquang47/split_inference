""" Module xử lý metrics FPS """
import time
import psutil
from prometheus_client import Gauge, Counter
import src.Log
import threading

class Metrics:
    def __init__(self, layer_id, client_id):
        self.layer_id = layer_id
        self.client_id = client_id
        
        # Initialize FPS metrics
        self.fps_metric = Gauge(
            f'layer_{layer_id}_fps_in_5s_window',
            f'Processing 5s window FPS for layer {layer_id}',
            ['client_id']
        )
        self.frame_counter = Counter(
            f'layer_{layer_id}_frames_processed_from_start',
            f'Total frames processed from start by layer {layer_id}',
            ['client_id']
        )
        self.batch_processing_time = Gauge(
            f'layer_{layer_id}_batch_processing_time_seconds',
            f'Processing time per BATCH for layer {layer_id}',
            ['client_id']
        )
        self.layer_processing_time = Counter(
            f'layer_{layer_id}_layer_processing_time_seconds',
            f'Total processing time for LAYERLAYER {layer_id}',
            ['client_id']
        )

        # Initialize queue metrics
        self.queue_length = Gauge(
            f'layer_{layer_id}_local_queue_length',
            f'Current length of local queue for layer {layer_id}',
            ['client_id']
        )

        # Initialize system metrics
        self.cpu_usage = Gauge(
            f'layer_{layer_id}_cpu_usage_percent',
            f'CPU usage percentage for layer {layer_id}',
            ['client_id']
        )
        self.ram_usage = Gauge(
            f'layer_{layer_id}_ram_usage_percent',
            f'RAM usage percentage for layer {layer_id}',
            ['client_id']
        )
        self.ram_usage_bytes = Gauge(
            f'layer_{layer_id}_ram_usage_bytes',
            f'RAM usage in bytes for layer {layer_id}',
            ['client_id']
        )
        self.ram_usage_mb = Gauge(
            f'layer_{layer_id}_ram_usage_mb',
            f'RAM usage in MB for layer {layer_id}',
            ['client_id']
        )

        # Initialize metrics with default values
        self.fps_metric.labels(client_id=self.client_id).set(0)
        self.frame_counter.labels(client_id=self.client_id).inc(0)
        self.batch_processing_time.labels(client_id=self.client_id).set(0)
        self.layer_processing_time.labels(client_id=self.client_id).inc(0)
        
        self.queue_length.labels(client_id=self.client_id).set(0)
        
        self.cpu_usage.labels(client_id=self.client_id).set(0)
        self.ram_usage.labels(client_id=self.client_id).set(0)
        self.ram_usage_bytes.labels(client_id=self.client_id).set(0)
        self.ram_usage_mb.labels(client_id=self.client_id).set(0)

        # Initialize counters
        self.window_start_time = time.time()
        self.window_frame_count = 0
        self.window_processing_time = 0
        self.total_frames = 0

        # Start system metrics collection in background
        self._start_system_metrics_collection()

    def _start_system_metrics_collection(self):
        """Start system metrics collection in background thread"""
        def collect_metrics():
            # Get the current process
            process = psutil.Process()
            
            while True:
                try:
                    # Get CPU usage for current process - using None interval to avoid blocking
                    cpu_percent = process.cpu_percent(interval=None)
                    self.cpu_usage.labels(client_id=self.client_id).set(cpu_percent)

                    # Get memory info for current process
                    memory_info = process.memory_info()
                    ram_percent = (memory_info.rss / psutil.virtual_memory().total) * 100
                    self.ram_usage.labels(client_id=self.client_id).set(ram_percent)
                    self.ram_usage_bytes.labels(client_id=self.client_id).set(memory_info.rss)
                    self.ram_usage_mb.labels(client_id=self.client_id).set(memory_info.rss / (1024 * 1024))  # Convert bytes to MB
                    
                except Exception as e:
                    src.Log.log_error(f"Error updating system metrics: {str(e)}")
                time.sleep(1)  # Update every 1 seconds - good balance between accuracy and resource usage

        # Start collection in background thread
        thread = threading.Thread(target=collect_metrics, daemon=True)
        thread.start()

    def update_fps_metrics(self, batch_frame, batch_processing_time, debug_mode=False):
        """Update FPS metrics"""
        current_time = time.time()
        self.window_frame_count += batch_frame
        self.window_processing_time += batch_processing_time
        self.total_frames += batch_frame

        if current_time - self.window_start_time >= 5:
            if self.window_processing_time > 0:
                window_fps = self.window_frame_count / self.window_processing_time
                src.Log.log_info(
                    f"Layer {self.layer_id} - FPS (5s window): {window_fps:.2f} - Total frames (from start): {self.total_frames}"
                )
                src.Log.log_info("Continue processing...")
                self.fps_metric.labels(client_id=self.client_id).set(window_fps)
            
            # Reset window counters
            self.window_start_time = current_time
            self.window_frame_count = 0
            self.window_processing_time = 0

    def update_batch_metrics(self, batch_frame, batch_processing_time):
        """Update batch processing metrics"""
        self.frame_counter.labels(client_id=self.client_id).inc(batch_frame)
        self.batch_processing_time.labels(client_id=self.client_id).set(batch_processing_time)
        self.layer_processing_time.labels(client_id=self.client_id).inc(batch_processing_time)

    def update_queue_length(self, length):
        """Update local queue length metric"""
        self.queue_length.labels(client_id=self.client_id).set(length)

    def get_fps_summary(self):
        """Get a summary of FPS metrics"""
        return {
            'total_frames': self.total_frames,
            'current_fps': self.fps_metric.labels(client_id=self.client_id)._value.get()
        } 