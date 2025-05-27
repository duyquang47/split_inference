""" Module xử lý metrics FPS """
import time
from prometheus_client import Gauge, Counter
import src.Log

class FPSMetrics:
    def __init__(self, layer_id, client_id):
        self.layer_id = layer_id
        self.client_id = client_id
        
        """ Initialize Prometheus metrics """
        self.fps_metric = Gauge(
            f'layer_{layer_id}_fps',
            f'Processing FPS for layer {layer_id}',
            ['client_id']
        )
        self.frame_counter = Counter(
            f'layer_{layer_id}_frames_processed',
            f'Total frames processed by layer {layer_id}',
            ['client_id']
        )
        self.processing_time = Gauge(
            f'layer_{layer_id}_processing_time_seconds',
            f'Processing time per batch for layer {layer_id}',
            ['client_id']
        )
        self.total_processing_time = Counter(
            f'layer_{layer_id}_total_processing_time_seconds',
            f'Total processing time for layer {layer_id}',
            ['client_id']
        )

        """ Initialize metrics with default values """
        self.fps_metric.labels(client_id=self.client_id).set(0)
        self.frame_counter.labels(client_id=self.client_id).inc(0)
        self.processing_time.labels(client_id=self.client_id).set(0)
        self.total_processing_time.labels(client_id=self.client_id).inc(0)

        """ Initialize 5-second window tracking """
        self.window_start_time = time.time()
        self.window_frame_count = 0
        self.window_processing_time = 0
        self.total_frames = 0

    def update_metrics(self, batch_frame, batch_processing_time, debug_mode=False):
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
            
            """ Reset window counters """
            self.window_start_time = current_time
            self.window_frame_count = 0
            self.window_processing_time = 0

    def update_batch_metrics(self, batch_frame, batch_processing_time):
        """ Update frame counter """
        self.frame_counter.labels(client_id=self.client_id).inc(batch_frame)
        
        """ Update processing time metrics """
        self.processing_time.labels(client_id=self.client_id).set(batch_processing_time)
        self.total_processing_time.labels(client_id=self.client_id).inc(batch_processing_time)
