# split_inference

## Configuration
Application configuration is in the `config.yaml` file:
```yaml
name: YOLO
server:
  cut-layer: a #or b, c
  clients:
    - 1
    - 1
  model: yolov8n
  batch-frame: 1
rabbit:
  address: 127.0.0.1
  username: admin
  password: admin
  virtual-host: /

data: video.mp4
log-path: .
control-count: 10
debug-mode: False
```
This configuration is use for server.

## How to Run
Alter your configuration, you need to run the server to listen and control the request from clients.
### Server
```commandline
python server.py
```
### Client
Now, when server is ready, run clients simultaneously with total number of client that you defined.

**Layer 1**

```commandline
python client.py --layer_id 1 
```
Where:
- `--layer_id` is the ID index of client's layer, start from 1.

If you want to use a specific device configuration for the training process, declare it with the `--device` argument when running the command line:
```commandline
python client.py --layer_id 1 --device cpu
```

## Result
Results include inference time, operating time, utilization. It locates in `result.log`.  
```text
============================
Total input frames: 3630
Total output frames: 3630
Frame drop rate: 0.00%
All time: 507.97s
Inference time: 489.24s
Utilization: 96.31%
Average FPS: 7.15
============================

```
