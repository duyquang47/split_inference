"""
Module khởi tạo client.
"""
import pika
import uuid
import argparse
import yaml
import sys
import torch

import src.Log
from src.RpcClient import RpcClient
from src.Scheduler import Scheduler

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Split learning framework")
    parser.add_argument('--layer_id', type=int, required=True, help='ID of layer, start from 1')
    parser.add_argument('--device', type=str, required=False, help='Device of client')
    return parser.parse_args()

def load_config():
    """Load and validate configuration file"""
    try:
        with open('config.yaml', 'r') as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        src.Log.log_error("config.yaml not found")
        sys.exit(1)
    except yaml.YAMLError as e:
        src.Log.log_error(f"Error parsing config.yaml: {e}")
        sys.exit(1)

def setup_device(device_arg):
    """Setup and validate device for client"""
    if device_arg is None:
        if torch.cuda.is_available():
            device = "cuda"
            src.Log.log_info(f"Using device: {torch.cuda.get_device_name(device)}")
        else:
            device = "cpu"
            src.Log.log_info("Using device: CPU")
    else:
        device = device_arg
        src.Log.log_info(f"Using device: CPU")
    return device

def setup_rabbitmq_connection(config):
    """Setup RabbitMQ connection"""
    try:
        credentials = pika.PlainCredentials(
            config["rabbit"]["username"],
            config["rabbit"]["password"]
        )
        connection = pika.BlockingConnection(
            pika.ConnectionParameters(
                config["rabbit"]["address"],
                5672,
                config["rabbit"]["virtual-host"],
                credentials
            )
        )
        channel = connection.channel()
        return channel
    except pika.exceptions.AMQPConnectionError as e:
        src.Log.log_error(f"Failed to connect to RabbitMQ: {e}")
        sys.exit(1)
    except Exception as e:
        src.Log.log_error(f"Unexpected error during connection setup: {e}")
        sys.exit(1)

def main():
    """Main function to initialize and start client"""
    try:
        """ Parse arguments and load configuration """
        args = parse_arguments()
        config = load_config()
        
        """ Setup device and generate client ID """
        device = setup_device(args.device)
        client_id = uuid.uuid4()
        
        """ Setup RabbitMQ connection """
        channel = setup_rabbitmq_connection(config)
        
        """ Initialize and start client """
        src.Log.log_info("Initializing client...")
        scheduler = Scheduler(client_id, args.layer_id, channel, device)
        client = RpcClient(
            client_id,
            args.layer_id,
            config["rabbit"]["address"],
            config["rabbit"]["username"],
            config["rabbit"]["password"],
            config["rabbit"]["virtual-host"],
            scheduler.inference_func,
            device
        )
        
        src.Log.log_info("Sending registration message to server...")
        client._setup_connection()
        client._register_with_server()
        client.wait_response()
        
    except KeyboardInterrupt:
        src.Log.log_info("Client shutdown initiated")
    except Exception as e:
        src.Log.log_error(f"Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
