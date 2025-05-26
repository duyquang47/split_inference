import os
import sys
import base64
import pika
import pickle
import torch
import torch.nn as nn
import threading

import src.Model
import src.Log

#công dụng server: thiết lập kết nối với client, đọc config, gửi config đến client
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

        #khởi tạo kết nối và channel
        credentials = pika.PlainCredentials(username, password)
        self.connection = pika.BlockingConnection(
            pika.ConnectionParameters(address, 5672, virtual_host, credentials)
        )
        #khởi tạo channel để nhận request
        self.channel = self.connection.channel()
        self.channel.queue_declare(queue='rpc_queue')
        self.channel.basic_qos(prefetch_count=1)
        self.channel.basic_consume(
            queue='rpc_queue', on_message_callback=self.on_request
        )
        #khởi tạo channel reply REGISTER client 
        self.reply_channel = self.connection.channel()

        self.register_clients = [0 for _ in range(len(self.total_clients))]
        self.list_clients = []
        self.list_clients_lock = threading.Lock()  # Đảm bảo thread-safe nếu cần

        self.data = config["data"]
        self.debug_mode = config["debug-mode"]
        log_path = config["log-path"]

        #khởi tạo log
        self.logger = src.Log.Logger(f"{log_path}/app.log")
        self.logger.log_info(
            f"Application start. Server is waiting for {self.total_clients} clients."
        )

    def start(self):
        #bắt đầu nhận message từ rpc_queue
        self.channel.start_consuming()

    def on_request(self, ch, method, props, body):
        #xử lý REGISTER message từ client
        message = pickle.loads(body)
        action = message.get("action")
        client_id = message.get("client_id")
        layer_id = message.get("layer_id")

        if action == "REGISTER":
            entry = (str(client_id), layer_id)
            with self.list_clients_lock:
                if entry not in self.list_clients:
                    self.list_clients.append(entry)
                    src.Log.print_with_color(
                        f"[<<<] Received REGISTER from client: {message}", "blue"
                    )
                    # Gửi START ngay cho client mới
                    self.notify_single_client(client_id, layer_id)
        ch.basic_ack(delivery_tag=method.delivery_tag)

    def notify_single_client(self, client_id, layer_id):
        #xác định điểm cắt, load model và gửi START đến client
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
        #gửi thông báo đến client
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

    #? chưa hiểu thực sự công dụng hàm này
    # Hàm gửi STOP cho tất cả client last layer (gọi khi muốn shutdown)
    def send_stop_to_last_layer(self, queue_name, last_layer_id):
        with self.list_clients_lock:
            num_last_layer_clients = sum(1 for cid, lid in self.list_clients if lid == last_layer_id)
        for _ in range(num_last_layer_clients):
            self.reply_channel.basic_publish(
                exchange='',
                routing_key=queue_name,
                body=pickle.dumps('STOP')
            )