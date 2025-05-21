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
from src.FrameMetrics import FrameMetrics

class Scheduler:
    def __init__(self, client_id, layer_id, channel, device):
        self.client_id = client_id
        self.layer_id = layer_id
        self.channel = channel
        self.device = device
        self.intermediate_queue = "intermediate_queue"
        self.channel.queue_declare(self.intermediate_queue, durable=False)

        # Initialize metrics
        self.fps_metrics = FPSMetrics(layer_id, client_id)
        self.frame_metrics = FrameMetrics(layer_id, client_id)

        logging.info(f"Initialized metrics for layer {layer_id} with client_id {client_id}")

        # Use fixed ports for each layer
        prometheus_ports = {
            1: 8001,
            2: 8002
        }

        if layer_id in prometheus_ports:
            prometheus_port = prometheus_ports[layer_id]
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.bind(('0.0.0.0', prometheus_port))  # Bind to all interfaces
                sock.close()
                #start server prometheus tại 2 port 8001 và 8002
                start_http_server(prometheus_port) 
                logging.info(f"Started Prometheus metrics server for layer {layer_id} on port {prometheus_port}")
            except OSError as e:
                if e.errno == 98:
                    logging.warning(f"Prometheus metrics server already running on port {prometheus_port} for layer {layer_id}")
                else:
                    logging.error(f"Failed to start Prometheus metrics server for layer {layer_id}: {e}")
                    raise
            except Exception as e:
                logging.error(f"Unexpected error starting Prometheus metrics server for layer {layer_id}: {e}")
                raise
        else:
            logging.warning(f"No Prometheus port configured for layer {layer_id}")

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

        model.eval() #set chế độ inference
        model.to(self.device) #đưa model vào device đọc ở command

        # Stream video directly from video server
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
                self.send_next_layer(self.intermediate_queue, 'STOP', logger)
                break

            # Update input frame count
            self.frame_metrics.update_input_frames(1)

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
                self.frame_metrics.update_output_frames(batch_frame)

                self.send_next_layer(self.intermediate_queue, y, logger)
                input_image = []
                pbar.update(batch_frame)

        cap.release()
        pbar.close()

        # Log frame metrics summary
        metrics_summary = self.frame_metrics.get_metrics_summary()
        logger.log_info(f"Frame Metrics Summary for Layer {self.layer_id}:")
        logger.log_info(f"Total Input Frames: {metrics_summary['input_frames']}")
        logger.log_info(f"Total Output Frames: {metrics_summary['output_frames']}")
        logger.log_info(f"Frame Drop Rate: {metrics_summary['drop_rate']}")
        logger.log_info(f"End Inference.")
        return time_inference

    def last_layer_callback(self, ch, method, properties, body):
        received_data = pickle.loads(body)
        if received_data != 'STOP':
            # Xử lý logic như trong while True cũ
            self.frame_metrics.update_input_frames(self.batch_frame)
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
            self.frame_metrics.update_output_frames(self.batch_frame)
            self.pbar.update(self.batch_frame)
        else:
            self.pbar.close()
            ch.stop_consuming()

    def last_layer(self, model, batch_frame, logger):
        self.model = model
        self.batch_frame = batch_frame
        self.logger = logger
        self.time_inference = 0
        self.total_processing_time = 0
        self.total_frames = 0
        self.pbar = tqdm(desc="Processing video (while loop)", unit="frame")

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

        # Sau khi stop_consuming, log metrics như cũ
        metrics_summary = self.frame_metrics.get_metrics_summary()
        logger.log_info(f"Frame Metrics Summary for Layer {self.layer_id}:")
        logger.log_info(f"Total Input Frames: {metrics_summary['input_frames']}")
        logger.log_info(f"Total Output Frames: {metrics_summary['output_frames']}")
        logger.log_info(f"Frame Drop Rate: {metrics_summary['drop_rate']}")
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