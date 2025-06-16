"""
Module thiết lập kết nối giữa server và client.
Cơ chế giao tiếp sử dụng là publish và consume message, trong đó:
- Client publish message REGISTER đến server để đăng ký
- Server consume message REGISTER từ client
- Server publish message START đến client, trong đó chứa config gửi đến client
- Client consume message START từ server, thực hiện load config và ghi log
"""
import pickle
import time
import base64
import os
from typing import Any, Dict, Optional, Callable

import pika
import torch
import torch.nn as nn
from ultralytics import YOLO

import src.Log
import src.Utils
from src.Model import SplitDetectionModel

class RpcClient:
    """ Khai báo các tham số của client """
    def __init__(
        self,
        client_id: str,
        layer_id: int,
        address: str,
        username: str,
        password: str,
        virtual_host: str,
        inference_func: Callable,
        device: str,
        prefetch_count: int = 1
    ):
        """ Gán giá trị cho các biến """
        self.client_id = client_id
        self.layer_id = layer_id
        self.address = address
        self.username = username
        self.password = password
        self.virtual_host = virtual_host
        self.inference_func = inference_func
        self.device = device
        self.prefetch_count = prefetch_count

        self.channel: Optional[pika.channel.Channel] = None
        self.connection: Optional[pika.connection.Connection] = None
        self.response: Optional[Dict[str, Any]] = None
        self.model: Optional[SplitDetectionModel] = None
        self.data: Optional[Any] = None
        self.debug_mode: bool = False

        self._setup_connection()
        self._register_with_server()

    def _setup_connection(self) -> None:
        """ Establish connection to RabbitMQ server """
        try:
            self.connection, self.channel = src.Utils.create_connection(
                self.address,
                self.username,
                self.password,
                self.virtual_host,
                self.prefetch_count
            )
        except Exception as e:
            src.Log.print_with_color(f"Failed to connect to RabbitMQ: {str(e)}", "red")
            raise

    def _register_with_server(self) -> None:
        """ Register client with the server """
        message = {
            "action": "REGISTER",
            "client_id": self.client_id,
            "layer_id": self.layer_id
        }
        self._send_to_server(message)

    def _send_to_server(self, message: Dict[str, Any]) -> None:
        try:
            self._setup_connection()
            self.channel.queue_declare('rpc_queue', durable=False)
            self.channel.basic_publish(
                exchange='',
                routing_key='rpc_queue',
                body=pickle.dumps(message)
            )
        except Exception as e:
            src.Log.print_with_color(f"Failed to send message to server: {str(e)}", "red")
            raise

    def _load_model(self, model_name: str, model_data: Optional[str]) -> None:
        """
        Load model from file or base64 encoded data.
        Args:
            model_name: Name of the model file
            model_data: Base64 encoded model data
        """
        file_path = f'{model_name}.pt'
        if model_data is not None:
            if os.path.exists(file_path):
                src.Log.print_with_color(f"Model file {file_path} already exists", "green")
            else:
                decoder = base64.b64decode(model_data)
                with open(file_path, "wb") as f:
                    f.write(decoder)
                src.Log.print_with_color(f"Loaded model to {file_path}", "green")
        else:
            src.Log.print_with_color("No model data provided", "yellow")

    def _handle_start_message(self, response: Dict[str, Any]) -> None:
        """
        Handle START message from server.
        Args:
            response: Server response containing configuration
        """
        model_name = response["model_name"]
        num_layers = response["num_layers"]
        splits = response["splits"]
        save_layers = response["save_layers"]
        batch_frame = response["batch_frame"]
        model_data = response["model"]
        data = response["data"]
        self.debug_mode = response["debug_mode"]

        self._load_model(model_name, model_data)

        pretrain_model = YOLO(f"{model_name}.pt").model
        self.model = SplitDetectionModel(pretrain_model, split_layer=splits)

        start_time = time.perf_counter()
        src.Log.print_with_color("Start Inference", "blue")
        
        inference_time = self.inference_func(
            self.model,
            data,
            num_layers,
            save_layers,
            batch_frame,
            self.debug_mode
        )
        
        total_time = time.perf_counter() - start_time

        src.Log.print_with_color("","blue")
        src.Log.print_with_color("============================", "blue")
        src.Log.print_with_color(f"All time: {total_time:.2f}s", "blue")
        src.Log.print_with_color(f"Inference time: {inference_time:.2f}s", "blue")
        src.Log.print_with_color(f"Utilization: {((inference_time / total_time) * 100):.2f}%", "blue")
        src.Log.print_with_color("============================", "blue")

    def wait_response(self) -> None:
        """ Wait for and process messages from server """
        reply_queue_name = f"reply_{self.client_id}"
        self.channel.queue_declare(reply_queue_name, durable=False)
        
        while True:
            method_frame, header_frame, body = self.channel.basic_get(
                queue=reply_queue_name,
                auto_ack=True
            )
            
            if body:
                response = pickle.loads(body)
                src.Log.log_info(
                    f"Client received: {response['message']}"
                )
                
                if response["action"] == "START":
                    self._handle_start_message(response)
                    break
            time.sleep(0.5)