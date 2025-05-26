#module định nghĩa chức năng mỗi layer và phương thức giao tiếp giữa các layer thông qua queue

import pickle
import time
from tqdm import tqdm
import torch
import cv2
import requests
from src.Model import SplitDetectionPredictor
from prometheus_client import start_http_server
import socket
import logging
import tempfile
import os
from src.FPS import FPSMetrics
import threading
import queue
import yaml

class Scheduler:
    def __init__(self, client_id, layer_id, channel, device):
        self.client_id = client_id
        self.layer_id = layer_id
        self.channel = channel
        self.device = device
        self.intermediate_queue = "intermediate_queue"
        self.channel.queue_declare(self.intermediate_queue, durable=False)

        # Initialize local queue
        self.local_queue = queue.Queue()

        # Initialize metrics
        self.fps_metrics = FPSMetrics(layer_id, client_id)

        prometheus_port = int(os.environ.get("PROMETHEUS_PORT", 0))
        if prometheus_port:
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.bind(('0.0.0.0', prometheus_port))  # Bind to all interfaces
                sock.close()
                #start http server với port cấu hình bên trên
                start_http_server(prometheus_port) 
                logging.info(f"Started Prometheus metrics server on port {prometheus_port}")
            except OSError as e:
                if e.errno == 98:
                    logging.warning(f"Prometheus metrics server already running on port {prometheus_port}")
                else:
                    logging.error(f"Failed to start Prometheus metrics server on port {prometheus_port}: {e}")
                    raise
            except Exception as e:
                logging.error(f"Unexpected error starting Prometheus metrics server on port {prometheus_port}: {e}")
                raise
        else:
            logging.warning(f"No Prometheus port configured for this client")

    #hàm mã hóa dữ liệu thành byte để gửi đi và giải mã trở vể sử dụng pickle
    #sử dụng cơ chế basic.publish() của queue
    def send_next_layer(self, intermediate_queue, data, logger):
        if data != 'STOP':
            data["layers_output"] = [t.cpu() if isinstance(t, torch.Tensor) else None for t in data["layers_output"]]
            message = pickle.dumps({
                "action": "OUTPUT",
                "data": data
            })
            self.channel.basic_publish(
                exchange='',
                routing_key=intermediate_queue,
                body=message,
            )
        else:
            message = pickle.dumps(data)
            self.channel.basic_publish(
                exchange='',
                routing_key=intermediate_queue,
                body=message,
            )

    def first_layer(self, model, data, save_layers, batch_frame, logger):
        time_inference = 0
        input_image = []
        predictor = SplitDetectionPredictor(model, overrides={"imgsz": 640})

        model.eval()
        model.to(self.device)

        #stream video từ video_server
        video_url = data
        cap = cv2.VideoCapture(video_url)
        if not cap.isOpened():
            logger.log_error(f"Could not open video stream from {video_url}")
            return False

        fps = cap.get(cv2.CAP_PROP_FPS)
        logger.log_info(f"FPS input: {fps}")
        pbar = tqdm(desc="Processing video (while loop)", unit="frame")

        total_frames = 0 #khởi tạo biến đếm frame
        total_processing_time = 0 #khởi tạo biến đếm thời gian

        while True:
            ret, frame = cap.read()
            if not ret:
                #gửi STOP nhiều hơn số pod scale dự kiến
                for _ in range(10):
                    self.send_next_layer(self.intermediate_queue, 'STOP', logger)
                break

            frame = cv2.resize(frame, (640, 640))
            tensor = torch.from_numpy(frame).float().permute(2, 0, 1)
            tensor /= 255.0
            input_image.append(tensor)

            if len(input_image) == batch_frame:
                start = time.perf_counter()

                input_image = torch.stack(input_image).to(self.device)
                predictor.setup_source(input_image)
                for predictor.batch in predictor.dataset:
                    path, input_image, _ = predictor.batch

                preprocess_image = predictor.preprocess(input_image)
                y = model.forward_head(preprocess_image, save_layers)

                batch_processing_time = time.perf_counter() - start
                time_inference += batch_processing_time
                total_processing_time += batch_processing_time
                total_frames += batch_frame

                # Update metrics
                self.fps_metrics.update_batch_metrics(batch_frame, batch_processing_time)
                self.fps_metrics.update_metrics(batch_frame, batch_processing_time, logger)

                self.send_next_layer(self.intermediate_queue, y, logger)
                input_image = []
                pbar.update(batch_frame)

        cap.release()
        pbar.close()

        logger.log_info(f"End Inference.")
        return time_inference

    def last_layer_callback(self, ch, method, properties, body):
        received_data = pickle.loads(body)
        self.local_queue.put(received_data)
        if received_data == 'STOP':
            self.pbar.close()
            ch.stop_consuming()

    def inference_worker(self):
        while True:
            received_data = self.local_queue.get()
            if received_data == 'STOP':
                break
            y = received_data["data"]
            y["layers_output"] = [t.to(self.device) if t is not None else None for t in y["layers_output"]]
            start = time.perf_counter()
            predictions = self.model.forward_tail(y)
            batch_processing_time = time.perf_counter() - start
            self.time_inference += batch_processing_time
            self.total_processing_time += batch_processing_time
            self.total_frames += self.batch_frame
            self.fps_metrics.update_batch_metrics(self.batch_frame, batch_processing_time)
            self.fps_metrics.update_metrics(self.batch_frame, batch_processing_time, self.logger)
            self.pbar.update(self.batch_frame)

    def last_layer(self, model, batch_frame, logger):
        self.model = model
        self.batch_frame = batch_frame
        self.logger = logger
        self.time_inference = 0
        self.total_processing_time = 0
        self.total_frames = 0
        self.pbar = tqdm(desc="Processing video (while loop)", unit="frame")
        self.local_queue = queue.Queue()

        # Start inference worker thread
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
        logger.log_info("Waiting for messages. To exit press CTRL+C")
        self.channel.start_consuming()
        worker.join()

        logger.log_info(f"End Inference.")
        return self.time_inference

    def middle_layer(self, model):
        pass

    def inference_func(self, model, data, num_layers, save_layers, batch_frame, logger):
        time_inference = 0
        if self.layer_id == 1:
            time_inference = self.first_layer(model, data, save_layers, batch_frame, logger)
        elif self.layer_id == num_layers:
            time_inference = self.last_layer(model, batch_frame, logger)
        else:
            self.middle_layer(model)
        return time_inference
