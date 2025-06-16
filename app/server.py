import argparse
import sys
import signal
from src.Server import Server
from src.Utils import delete_old_queues
import src.Log
import yaml

def load_config():
    try:
        with open('config.yaml') as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        src.Log.log_error("config.yaml not found")
        sys.exit(1)
    except yaml.YAMLError as e:
        src.Log.log_error(f"Error parsing config.yaml: {e}")
        sys.exit(1)

def signal_handler(sig, frame):
    src.Log.log_info("Received stop signal (Ctrl+C). Cleaning up...")
    try:
        delete_old_queues(address, username, password, virtual_host, prefetch_count)
        src.Log.log_success("Cleanup completed successfully")
    except Exception as e:
        src.Log.log_error(f"Error during cleanup: {e}")
    sys.exit(0)

if __name__ == "__main__":
    try:
        config = load_config()
        address = config["rabbit"]["address"]
        username = config["rabbit"]["username"]
        password = config["rabbit"]["password"]
        virtual_host = config["rabbit"]["virtual-host"]
        prefetch_count = config["rabbit"]["prefetch-count"]

        signal.signal(signal.SIGINT, signal_handler)
        delete_old_queues(address, username, password, virtual_host, prefetch_count)
        
        server = Server(config)
        server.start()
        src.Log.log_info("Server is ready and waiting for clients")
    except Exception as e:
        src.Log.log_error(f"Failed to start server: {e}")
        sys.exit(1)
