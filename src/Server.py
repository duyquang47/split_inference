"""
Module xử lý server.
Công dụng: thiết lập kết nối với client, đọc config, gửi config đến client
"""
import os
import sys
import base64
import pika
import pickle
import torch
import torch.nn as nn
import threading
import time

import src.Model
import src.Log

class Server:
    def __init__(self, config):
        try:
            self._load_config(config)
            self._setup_connection()
            self._initialize_metrics()
            src.Log.log_success(
                f"Server initialized successfully. Waiting for {self.total_clients} clients."
            )
        except Exception as e:
            src.Log.log_error(f"Failed to initialize server: {e}")
            raise

    def _load_config(self, config):
        """Load and validate configuration"""
        self.model_name = config["server"]["model"]
        self.total_clients = config["server"]["clients"]
        self.cut_layer = config["server"]["cut-layer"]
        self.batch_frame = config["server"]["batch-frame"]
        self.data = config["data"]
        self.debug_mode = config["debug-mode"]

        """ RabbitMQ configuration """
        self.rabbit_config = {
            "address": config["rabbit"]["address"],
            "username": config["rabbit"]["username"],
            "password": config["rabbit"]["password"],
            "virtual_host": config["rabbit"]["virtual-host"]
        }

    def _setup_connection(self):
        """Setup RabbitMQ connection and channels"""
        try:
            credentials = pika.PlainCredentials(
                self.rabbit_config["username"],
                self.rabbit_config["password"]
            )
            self.connection = pika.BlockingConnection(
                pika.ConnectionParameters(
                    self.rabbit_config["address"],
                    5672,
                    self.rabbit_config["virtual_host"],
                    credentials
                )
            )
            
            """ Main channel for receiving requests """
            self.channel = self.connection.channel()
            self.channel.queue_declare(queue='rpc_queue')
            self.channel.basic_qos(prefetch_count=1)
            self.channel.basic_consume(
                queue='rpc_queue',
                on_message_callback=self.on_request
            )
            
            """ Reply channel for sending responses """
            self.reply_channel = self.connection.channel()
            
        except pika.exceptions.AMQPConnectionError as e:
            src.Log.log_error(f"Failed to connect to RabbitMQ: {e}")
            raise
        except Exception as e:
            src.Log.log_error(f"Unexpected error during connection setup: {e}")
            raise

    def _initialize_metrics(self):
        """Initialize tracking variables"""
        self.register_clients = [0 for _ in range(len(self.total_clients))]
        self.list_clients = []
        self.list_clients_lock = threading.Lock()
        self.start_time = time.time()

    def start(self):
        """Start the server and begin consuming messages"""
        try:
            self.channel.start_consuming()
        except KeyboardInterrupt:
            src.Log.log_info("Server shutdown initiated")
        except Exception as e:
            src.Log.log_error(f"Error during server operation: {e}")
        finally:
            self._cleanup()

    def _cleanup(self):
        """Cleanup resources"""
        try:
            if hasattr(self, 'connection') and self.connection.is_open:
                self.connection.close()
            src.Log.log_success("Server cleanup completed")
        except Exception as e:
            src.Log.log_error(f"Error during cleanup: {e}")

    def on_request(self, ch, method, props, body):
        """Handle incoming registration requests"""
        try:
            message = pickle.loads(body)
            action = message.get("action")
            client_id = message.get("client_id")
            layer_id = message.get("layer_id")

            if action == "REGISTER":
                self._handle_registration(client_id, layer_id, message)
            else:
                src.Log.log_warning(f"Received unknown action: {action}")

            ch.basic_ack(delivery_tag=method.delivery_tag)
        except Exception as e:
            src.Log.log_error(f"Error processing request: {e}")
            ch.basic_nack(delivery_tag=method.delivery_tag, requeue=True)

    def _handle_registration(self, client_id, layer_id, message):
        """Handle client registration"""
        entry = (str(client_id), layer_id)
        with self.list_clients_lock:
            if entry not in self.list_clients:
                self.list_clients.append(entry)
                src.Log.log_success(
                    f"Client registered - ID: {client_id}, Layer: {layer_id}"
                )
                self.notify_single_client(client_id, layer_id)

    def notify_single_client(self, client_id, layer_id):
        """Send start notification to a client"""
        try:
            model_data = self._load_model()
            response = self._prepare_start_response(model_data, layer_id)
            self.send_to_response(client_id, pickle.dumps(response))
        except Exception as e:
            src.Log.log_error(f"Failed to notify client {client_id}: {e}")
            raise

    def _load_model(self):
        """Load and encode model file"""
        file_path = f"{self.model_name}.pt"
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Model file {file_path} not found")

        src.Log.log_info(f"Loading model {self.model_name}")
        with open(file_path, "rb") as f:
            return base64.b64encode(f.read()).decode('utf-8')

    def _prepare_start_response(self, model_data, layer_id):
        """Prepare start response for client"""
        default_splits = {
            "a": (10, [4, 6, 9]),
            "b": (16, [9, 12, 15]),
            "c": (22, [15, 18, 21])
        }
        splits = default_splits[self.cut_layer]

        return {
            "action": "START",
            "message": "Server accept the connection",
            "model": model_data,
            "splits": splits[0],
            "save_layers": splits[1],
            "batch_frame": self.batch_frame,
            "num_layers": len(self.total_clients),
            "model_name": self.model_name,
            "data": self.data,
            "debug_mode": self.debug_mode
        }

    def send_to_response(self, client_id, message):
        """Send message to client's reply queue"""
        try:
            reply_queue = f"reply_{client_id}"
            self.reply_channel.queue_declare(reply_queue, durable=False)
            self.reply_channel.basic_publish(
                exchange='',
                routing_key=reply_queue,
                body=message
            )
            src.Log.log_info(f"Response sent to client {client_id}")
        except Exception as e:
            src.Log.log_error(f"Failed to send response to client {client_id}: {e}")
            raise

    """ Not needed """
    def send_stop_to_last_layer(self, queue_name, last_layer_id):
        """Send stop signal to all last layer clients"""
        try:
            with self.list_clients_lock:
                num_last_layer_clients = sum(
                    1 for cid, lid in self.list_clients if lid == last_layer_id
                )
            
            for _ in range(num_last_layer_clients):
                self.reply_channel.basic_publish(
                    exchange='',
                    routing_key=queue_name,
                    body=pickle.dumps('STOP')
                )
            src.Log.log_info(f"Stop signal sent to {num_last_layer_clients} last layer clients")
        except Exception as e:
            src.Log.log_error(f"Failed to send stop signal: {e}")
            raise