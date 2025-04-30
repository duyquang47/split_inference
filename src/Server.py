import os
import sys
import base64
import pika
import pickle
import torch
import torch.nn as nn

import src.Model
import src.Log

#cong dung server: doc config, gui config den client, ghi log
class Server:
    def __init__(self, config):
        address = config["rabbit"]["address"]
        username = config["rabbit"]["username"]
        password = config["rabbit"]["password"]
        virtual_host = config["rabbit"]["virtual-host"]

        self.model_name = config["server"]["model"]
        self.total_clients = config["server"]["clients"]
        self.cut_layer = config["server"]["cut-layer"]
        self.batch_frame = config["server"]["batch-frame"]

        #khoi tao ket noi va channel
        credentials = pika.PlainCredentials(username, password)
        self.connection = pika.BlockingConnection(
            pika.ConnectionParameters(address, 5672, virtual_host, credentials)
        )
        #khoi tao channel de nhan request
        self.channel = self.connection.channel()
        self.channel.queue_declare(queue='rpc_queue')
        self.channel.basic_qos(prefetch_count=1)
        self.channel.basic_consume(
            queue='rpc_queue', on_message_callback=self.on_request
        )
        #khoi tao channel de reply client
        self.reply_channel = self.connection.channel()

        self.register_clients = [0 for _ in range(len(self.total_clients))]
        self.list_clients = []

        self.data = config["data"]
        self.debug_mode = config["debug-mode"]
        log_path = config["log-path"]

        #khoi tao log
        self.logger = src.Log.Logger(f"{log_path}/app.log")
        self.logger.log_info(
            f"Application start. Server is waiting for {self.total_clients} clients."
        )

    def start(self):
        #bat dau nhan message tu rpc_queue
        self.channel.start_consuming()

    def on_request(self, ch, method, props, body):
        #xu ly REGISTER message tu client
        message = pickle.loads(body)
        action = message.get("action")
        client_id = message.get("client_id")
        layer_id = message.get("layer_id")

        if action == "REGISTER":
            #dang ky client neu chua co
            entry = (str(client_id), layer_id)
            if entry not in self.list_clients:
                self.list_clients.append(entry)

            src.Log.print_with_color(
                f"[<<<] Received message from client: {message}", "blue"
            )
            #tang bo dem cho client
            self.register_clients[layer_id - 1] += 1

            #du client, ghi log
            if self.register_clients == self.total_clients:
                src.Log.print_with_color(
                    "All clients are connected. Sending notifications.", "green"
                )
                self.notify_clients()

        ch.basic_ack(delivery_tag=method.delivery_tag)

    def notify_clients(self):
        #xac dinh diem cat, load model va gui START den client
        default_splits = {
            "a": (10, [4, 6, 9]),
            "b": (16, [9, 12, 15]),
            "c": (22, [15, 18, 21])
        }
        splits = default_splits[self.cut_layer]
        file_path = f"{self.model_name}.pt"

        if os.path.exists(file_path):
            src.Log.print_with_color(
                f"Load model {self.model_name}.", "green"
            )
            with open(file_path, "rb") as f:
                file_bytes = f.read()
                encoded = base64.b64encode(file_bytes).decode('utf-8')
        else:
            src.Log.print_with_color(
                f"{self.model_name} does not exist.", "yellow"
            )
            sys.exit(1)

        for client_id, layer_id in self.list_clients:
            response = {
                "action": "START",
                "message": "Server accept the connection",
                "model": encoded,
                "splits": splits[0],
                "save_layers": splits[1],
                "batch_frame": self.batch_frame,
                "num_layers": len(self.total_clients),
                "model_name": self.model_name,
                "data": self.data,
                "debug_mode": self.debug_mode
            }
            self.send_to_response(client_id, pickle.dumps(response))

    def send_to_response(self, client_id, message):
        #gui config da load den cac client thong qua queue rieng
        reply_queue = f"reply_{client_id}"
        self.reply_channel.queue_declare(reply_queue, durable=False)
        src.Log.print_with_color(
            f"[>>>] Sent notification to client {client_id}", "red"
        )
        self.reply_channel.basic_publish(
            exchange='',
            routing_key=reply_queue,
            body=message
        )