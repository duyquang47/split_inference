""" Module định nghĩa chức năng mỗi layer và phương thức giao tiếp giữa các layer thông qua queue """

import pickle
import time
import torch
import cv2
import requests
from src.Model import SplitDetectionPredictor
from prometheus_client import start_http_server
import socket
import logging
import tempfile
import os
from src.Metrics import Metrics
import threading
import queue
import yaml
import src.Log

class Scheduler:
    def __init__(self, client_id, layer_id, channel, device):
        self.client_id = client_id
        self.layer_id = layer_id
        self.channel = channel
        self.device = device
        self.intermediate_queue = "intermediate_queue"
        self.channel.queue_declare(self.intermediate_queue, durable=False)
        self.local_queue = queue.Queue()
        self.metrics = Metrics(layer_id, client_id)
        self._setup_prometheus()

    def _setup_prometheus(self):
        prometheus_port = int(os.environ.get("PROMETHEUS_PORT", 0))
        if not prometheus_port:
            src.Log.log_warning("No Prometheus port configured for this client")
            return

        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.bind(('0.0.0.0', prometheus_port))
            sock.close()
            start_http_server(prometheus_port)
            src.Log.log_info(f"Started Prometheus metrics server on port {prometheus_port}")
        except OSError as e:
            if e.errno == 98:
                src.Log.log_warning(f"Prometheus metrics server already running on port {prometheus_port}")
            else:
                src.Log.log_error(f"Failed to start Prometheus metrics server: {e}")
                raise
        except Exception as e:
            src.Log.log_error(f"Unexpected error starting Prometheus server: {e}")
            raise

    def _prepare_message(self, data):
        if data != 'STOP':
            data["layers_output"] = [t.cpu() if isinstance(t, torch.Tensor) else None for t in data["layers_output"]]
            return pickle.dumps({
                "action": "OUTPUT",
                "data": data
            })
        return pickle.dumps(data)

    def send_next_layer(self, intermediate_queue, data, debug_mode=False):
        message = self._prepare_message(data)
        self.channel.basic_publish(
            exchange='',
            routing_key=intermediate_queue,
            body=message,
        )

    def _process_frame(self, frame):
        frame = cv2.resize(frame, (640, 640))
        tensor = torch.from_numpy(frame).float().permute(2, 0, 1)
        tensor /= 255.0
        return tensor

    def _process_batch(self, model, predictor, input_image, save_layers):
        start = time.perf_counter()
        
        input_image = torch.stack(input_image).to(self.device)
        predictor.setup_source(input_image)
        for predictor.batch in predictor.dataset:
            path, input_image, _ = predictor.batch

        preprocess_image = predictor.preprocess(input_image)
        y = model.forward_head(preprocess_image, save_layers)
        
        batch_processing_time = time.perf_counter() - start
        return y, batch_processing_time

    def _update_metrics(self, batch_frame, batch_processing_time, debug_mode=False):
        self.metrics.update_batch_metrics(batch_frame, batch_processing_time)
        self.metrics.update_fps_metrics(batch_frame, batch_processing_time, debug_mode)

    def first_layer(self, model, data, save_layers, batch_frame, debug_mode=False):
        src.Log.log_info(f"[Layer {self.layer_id}] Starting first layer processing with batch size {batch_frame}")
        time_inference = 0
        input_image = []
        predictor = SplitDetectionPredictor(model, overrides={"imgsz": 640})

        model.eval()
        model.to(self.device)

        cap = cv2.VideoCapture(data)
        if not cap.isOpened():
            src.Log.log_error(f"[Layer {self.layer_id}] Could not open video stream from {data}")
            return False

        fps = cap.get(cv2.CAP_PROP_FPS)
        src.Log.log_info(f"[Layer {self.layer_id}] Video FPS: {fps}")

        total_frames = 0
        total_processing_time = 0
        batches_processed = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    src.Log.log_info(f"[Layer {self.layer_id}] End of video stream. Total frames processed: {total_frames}")
                    for _ in range(10):
                        self.send_next_layer(self.intermediate_queue, 'STOP', debug_mode)
                    break

                input_image.append(self._process_frame(frame))

                if len(input_image) == batch_frame:
                    batches_processed += 1
                    src.Log.log_debug(f"[Layer {self.layer_id}] Processing batch {batches_processed}")
                    y, batch_processing_time = self._process_batch(model, predictor, input_image, save_layers)
                    
                    time_inference += batch_processing_time
                    total_processing_time += batch_processing_time
                    total_frames += batch_frame

                    # Update metrics for each batch
                    self._update_metrics(batch_frame, batch_processing_time, debug_mode)
                    
                    if debug_mode:
                        src.Log.log_debug(f"[Layer {self.layer_id}] Batch {batches_processed} processed in {batch_processing_time:.3f}s")
                    
                    self.send_next_layer(self.intermediate_queue, y, debug_mode)
                    input_image = []

        finally:
            cap.release()

        src.Log.log_info(f"[Layer {self.layer_id}] End Inference.")
        return time_inference

    def _process_received_data(self, received_data):
        y = received_data["data"]
        y["layers_output"] = [t.to(self.device) if t is not None else None for t in y["layers_output"]]
        return y

    def last_layer_callback(self, ch, method, properties, body):
        received_data = pickle.loads(body)
        self.local_queue.put(received_data)
        # Update queue length metric
        self.metrics.update_queue_length(self.local_queue.qsize())
        if received_data == 'STOP':
            ch.stop_consuming()

    def inference_worker(self):
        while True:
            received_data = self.local_queue.get()
            # Update queue length metric after getting item
            self.metrics.update_queue_length(self.local_queue.qsize())
            
            if received_data == 'STOP':
                break

            y = self._process_received_data(received_data)
            
            start = time.perf_counter()
            predictions = self.model.forward_tail(y)
            batch_processing_time = time.perf_counter() - start
            
            self.time_inference += batch_processing_time
            self.total_processing_time += batch_processing_time
            self.total_frames += self.batch_frame
            
            # Update metrics for each batch
            self._update_metrics(self.batch_frame, batch_processing_time, self.debug_mode)

    def last_layer(self, model, batch_frame, debug_mode=False):
        self.model = model
        self.batch_frame = batch_frame
        self.debug_mode = debug_mode
        self.time_inference = 0
        self.total_processing_time = 0
        self.total_frames = 0

        worker = threading.Thread(target=self.inference_worker)
        worker.start()

        last_queue = "intermediate_queue"
        self.channel.queue_declare(queue=last_queue, durable=False)
        self.channel.basic_qos(prefetch_count=50)
        self.channel.basic_consume(
            queue=last_queue,
            on_message_callback=self.last_layer_callback,
            auto_ack=True
        )

        src.Log.log_info("Waiting for messages. To exit press CTRL+C")
        self.channel.start_consuming()
        worker.join()

        src.Log.log_info("End Inference.")
        return self.time_inference

    def inference_func(self, model, data, num_layers, save_layers, batch_frame, debug_mode=False):
        time_inference = 0
        if self.layer_id == 1:
            time_inference = self.first_layer(model, data, save_layers, batch_frame, debug_mode)
        elif self.layer_id == num_layers:
            time_inference = self.last_layer(model, batch_frame, debug_mode)
        else:
            self.middle_layer(model)
        return time_inference