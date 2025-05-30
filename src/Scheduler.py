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
from src.FPS import FPSMetrics
import threading
import queue
import yaml
import pika
import src.Log
from src.FrameMetrics import FrameMetrics

class Scheduler:
    def __init__(self, client_id, layer_id, channel, device):
        self.client_id = client_id
        self.layer_id = layer_id
        self.channel = channel
        self.device = device
        self.intermediate_queue = "intermediate_queue"
        self.channel.queue_declare(self.intermediate_queue, durable=False)
        self.local_queue = queue.Queue()
        self.fps_metrics = FPSMetrics(layer_id, client_id)
        self.frame_metrics = None  # Will be initialized in inference_func
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
        try:
            self.channel.basic_publish(
                exchange='',
                routing_key=intermediate_queue,
                body=message,
            )
        except Exception as e:
            src.Log.log_error(f"Error publishing to {intermediate_queue_name}: {e}")

    def _process_frame(self, frame):
        frame = cv2.resize(frame, (640, 640))
        tensor = torch.from_numpy(frame).float().permute(2, 0, 1)
        tensor /= 255.0
        return tensor

    def _process_batch(self, model, predictor, input_image_list, save_layers):
        start = time.perf_counter() 
        
        if not input_image_list:
            src.Log.log_warning("Empty!")
            return None, 0

        stacked_input_image = torch.stack(input_image_list).to(self.device)
        predictor.setup_source(stacked_input_image)
        for predictor.batch in predictor.dataset:
            path, input_image, _ = predictor.batch

        preprocess_image = predictor.preprocess(stacked_input_image)
        y = model.forward_head(preprocess_image, save_layers)
        
        batch_processing_time = time.perf_counter() - start
        return y, batch_processing_time

    def _update_metrics(self, batch_frame, batch_processing_time, debug_mode=False):
        self.fps_metrics.update_batch_metrics(batch_frame, batch_processing_time)
        self.fps_metrics.update_metrics(batch_frame, batch_processing_time, debug_mode)

    def first_layer(self, model, data, save_layers, batch_frame, debug_mode=False):
        time_inference = 0
        input_image = []
        predictor = SplitDetectionPredictor(model, overrides={"imgsz": 640})

        model.eval()
        model.to(self.device)

        cap = cv2.VideoCapture(data)
        if not cap.isOpened():
            src.Log.log_error(f"Could not open video stream from {data}")
            return False

        fps = cap.get(cv2.CAP_PROP_FPS)
        src.Log.log_info(f"FPS input: {fps}")

        total_frames = 0
        total_processing_time = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    for _ in range(10):
                        self.send_next_layer(self.intermediate_queue, 'STOP', debug_mode)
                    src.Log.log_info("Input stream ended. Sent STOP signals.")
                    break

                input_image.append(self._process_frame(frame))

                if len(input_image) == batch_frame:
                    y, batch_processing_time = self._process_batch(model, predictor, input_image, save_layers)
                    
                    time_inference += batch_processing_time
                    total_processing_time += batch_processing_time
                    total_frames += batch_frame

                    # Update metrics for each batch
                    if self.frame_metrics:
                        self.frame_metrics.update_input_frames(batch_frame)
                    
                    self._update_metrics(batch_frame, batch_processing_time, debug_mode)
                    
                    self.send_next_layer(self.intermediate_queue, y, debug_mode)
                    
                    if self.frame_metrics:
                        self.frame_metrics.update_input_frames(batch_frame)
                    
                    input_image = []
        except Exception as e:
            src.Log.log_error(f"Error in first_layer processing loop: {e}")
        finally:
            cap.release()

        src.Log.log_info("End Inference for Layer 1")
        return time_inference

    def _process_received_data(self, received_data_payload):
        if isinstance(received_data_payload, dict) and received_data_payload.get("action") == "OUTPUT":
            y = received_data_payload["data"]
            if isinstance(y, dict) and "layers_output" in y and isinstance(y["layers_output"], list):
                y["layers_output"] = [t.to(self.device) if t is not None else None for t in y["layers_output"]]
            return y
        elif received_data_payload == 'STOP':
            return 'STOP'
        else:
            src.Log.log_warning(f"Received unexpected data format: {type(received_data_payload)}")
            return None 

    def last_layer_callback(self, ch, method, properties, body):
        delivery_tag = method.delivery_tag
        try: 
            received_data_payload = pickle.loads(body)
            item_for_worker = {
                "payload": received_data_payload,
                "delivery_tag": delivery_tag,
                "channel": ch  # Pass the callback channel to the worker
            }
            self.local_queue.put(item_for_worker)
            src.Log.log_debug(f"L2_CALLBACK: Queued item with tag {delivery_tag}.")
        
            if received_data_payload == 'STOP':
                src.Log.log_info(f"L2_CALLBACK: STOP signal received (tag: {delivery_tag}), consuming will be stopped by inference_worker sending ACK for STOP.")

        except pickle.UnpicklingError as e:
            src.Log.log_error(f"L2_CALLBACK: Failed to unpickle message (tag: {delivery_tag}): {e}. NACKing.")
            ch.basic_nack(delivery_tag=delivery_tag, requeue=False)
        except Exception as e:
            src.Log.log_error(f"L2_CALLBACK: Error in callback (tag: {delivery_tag}): {e}. NACKing.")
            ch.basic_nack(delivery_tag=delivery_tag, requeue=False)
    
    def inference_worker(self):
        src.Log.log_info("L2_WORKER: Inference worker started.")
        
        while True:
            queue_item = None
            delivery_tag = None
            callback_channel = None
            try:
                queue_item = self.local_queue.get("""timeout=1.0""")

                received_payload = queue_item["payload"]
                delivery_tag = queue_item["delivery_tag"]
                callback_channel = queue_item["channel"]  # Get the callback channel
                src.Log.log_debug(f"L2_WORKER: Dequeued item with tag {delivery_tag}.")
                    
                if received_payload == 'STOP':
                    src.Log.log_info(f"L2_WORKER: STOP signal processed (tag: {delivery_tag}). Acknowledging and preparing to stop consumer.")
                    if callback_channel and callback_channel.is_open:
                        try:
                            callback_channel.basic_ack(delivery_tag=delivery_tag)
                            src.Log.log_info(f"L2_WORKER: STOP message (tag: {delivery_tag}) ACKed.")
                        except Exception as e:
                            src.Log.log_error(f"L2_WORKER: Error ACK-ing STOP (tag: {delivery_tag}): {e}")
                    break

                y = self._process_received_data(received_payload)
                if y is None: # Error during _process_received_data or unexpected format
                    src.Log.log_warning(f"L2_WORKER: Invalid data (tag: {delivery_tag}). NACKing.")
                    if delivery_tag and callback_channel and callback_channel.is_open:
                        callback_channel.basic_nack(delivery_tag=delivery_tag, requeue=False)
                    continue
                
                start = time.perf_counter()
                predictions = self.model.forward_tail(y)
                batch_processing_time = time.perf_counter() - start
                
                self.time_inference += batch_processing_time
                self.total_processing_time += batch_processing_time
                self.total_frames += self.batch_frame
                
                # Update metrics for each batch
                if self.frame_metrics:
                    self.frame_metrics.update_output_frames(self.batch_frame)
                    self.frame_metrics.update_input_frames(self.batch_frame)
                
                self._update_metrics(self.batch_frame, batch_processing_time, self.debug_mode)
                src.Log.log_debug(f"L2_WORKER: Inference complete for tag {delivery_tag}. Processing time: {batch_processing_time:.4f}s")

                if callback_channel and callback_channel.is_open:
                    callback_channel.basic_ack(delivery_tag=delivery_tag)
                    src.Log.log_info(f"L2_WORKER: Message (tag: {delivery_tag}) ACKed successfully.")

            except queue.Empty:
                continue
            except Exception as e:
                src.Log.log_error(f"L2_WORKER: Error processing item (tag: {delivery_tag}): {e}")
                if delivery_tag and callback_channel and callback_channel.is_open:
                    try:
                        src.Log.log_warning(f"L2_WORKER: NACKing message (tag: {delivery_tag}) due to error.")
                        callback_channel.basic_nack(delivery_tag=delivery_tag, requeue=False)
                    except Exception as nack_e:
                        src.Log.log_error(f"L2_WORKER: Error sending NACK for tag {delivery_tag}: {nack_e}")               
        src.Log.log_info("L2_WORKER: Inference worker finished.")    
    
    def last_layer(self, model, batch_frame, debug_mode=False):
        self.model = model
        self.batch_frame = batch_frame
        self.debug_mode = debug_mode
        self.time_inference = 0
        self.total_processing_time = 0
        self.total_frames = 0

        if self.model and hasattr(self.model, 'to'):
            self.model.to(self.device)
        if self.model and hasattr(self.model, 'eval'):
            self.model.eval()
    
        worker = threading.Thread(target=self.inference_worker, name=f"InferenceWorker" )
        worker.daemon = True
        worker.start()

        local_queue = "local_queue" #################################### chu y doan nay
        try:
            self.channel.queue_declare(queue=local_queue, durable=False)
            self.channel.basic_qos(prefetch_count=50)
            self.channel.basic_consume(
                queue=local_queue,
                on_message_callback=self.last_layer_callback,
                auto_ack=False
            )

            src.Log.log_info("L2_MAIN: Waiting for messages. To exit press CTRL+C")
            self.channel.start_consuming()

        except pika.exceptions.StreamLostError as e:
            src.Log.log_error(f"L2_MAIN: RabbitMQ StreamLostError: {e}. Consumer loop will exit.")
        except KeyboardInterrupt:
            src.Log.log_info("L2_MAIN: KeyboardInterrupt received. Stopping consumption...")
            if self.channel and self.channel.is_open:
                self.channel.stop_consuming()
        except Exception as e:
            src.Log.log_error(f"L2_MAIN: Error during consumption: {e}")
            if self.channel and self.channel.is_open:
                self.channel.stop_consuming() # Attempt to stop consuming on other errors too
        finally:
            src.Log.log_info("L2_MAIN: Consumption loop finished or exited.")
            self.local_queue.put(
                {"payload": "STOP", 
                "delivery_tag": None,
                "channel": None}
            )
            src.Log.log_info("L2_MAIN: Waiting for inference worker to join...")
            worker.join(timeout=5.0) # Wait for worker to finish
            if worker.is_alive():
                src.Log.log_warning("L2_MAIN: Inference worker did not join in time.")
        
        src.Log.log_info("End Inference for Layer 2.")
        return self.time_inference

    def inference_func(self, model, data, num_layers, save_layers, batch_frame, debug_mode=False, frame_metrics=None):
        time_inference = 0
        self.frame_metrics = frame_metrics  # Use the frame_metrics passed from RpcClient
        if self.layer_id == 1:
            src.Log.log_info(f"Executing as FIRST layer (Layer ID: {self.layer_id})")
            time_inference = self.first_layer(model, data, save_layers, batch_frame, debug_mode)
        elif self.layer_id == num_layers:
            src.Log.log_info(f"Executing as LAST layer (Layer ID: {self.layer_id})")
            time_inference = self.last_layer(model, batch_frame, debug_mode)
        return time_inference