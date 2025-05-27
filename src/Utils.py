""" Module chứa các utility functions cho việc xử lý queue và kết nối """

import pika
from requests.auth import HTTPBasicAuth
import requests
from typing import Tuple, Optional

def create_connection(
    address: str,
    username: str,
    password: str,
    virtual_host: str
) -> Tuple[pika.BlockingConnection, pika.channel.Channel]:

    """ Tạo kết nối đến RabbitMQ server """
    credentials = pika.PlainCredentials(username, password)
    connection = pika.BlockingConnection(
        pika.ConnectionParameters(
            address,
            5672,
            virtual_host,
            credentials
        )
    )
    channel = connection.channel()
    """ Return tupple chứa connection và channel """
    return connection, channel

def delete_old_queues(address: str, username: str, password: str, virtual_host: str) -> bool:
    url = f'http://{address}:15672/api/queues'
    response = requests.get(url, auth=HTTPBasicAuth(username, password))

    if response.status_code == 200:
        queues = response.json()
        connection, channel = create_connection(address, username, password, virtual_host)

        for queue in queues:
            queue_name = queue['name']
            if queue_name.startswith("reply") or queue_name.startswith("intermediate_queue") or queue_name.startswith(
                    "result") or queue_name.startswith("rpc_queue"):
                channel.queue_delete(queue=queue_name)
            else:
                channel.queue_purge(queue=queue_name)

        connection.close()
        return True
    return False
