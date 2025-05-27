from prometheus_client import Counter, Gauge

class FrameMetrics:
    def __init__(self, layer_id, client_id):
        self.layer_id = layer_id
        self.client_id = client_id
        
        """ Initialize Prometheus metrics """
        self.input_frames = Counter(
            f'layer_{layer_id}_input_frames',
            f'Total input frames for layer {layer_id}',
            ['client_id']
        )
        
        self.output_frames = Counter(
            f'layer_{layer_id}_output_frames',
            f'Total output frames for layer {layer_id}',
            ['client_id']
        )
        
        self.frame_drop_rate = Gauge(
            f'layer_{layer_id}_frame_drop_rate',
            f'Frame drop rate for layer {layer_id}',
            ['client_id']
        )
        
        """ Initialize counters """
        self.total_input_frames = 0
        self.total_output_frames = 0

    def update_input_frames(self, batch_size):
        """Update input frame count"""
        self.total_input_frames += batch_size
        self.input_frames.labels(client_id=self.client_id).inc(batch_size)

    def update_output_frames(self, batch_size):
        """Update output frame count"""
        self.total_output_frames += batch_size
        self.output_frames.labels(client_id=self.client_id).inc(batch_size)

    def calculate_drop_rate(self):
        """Calculate and update frame drop rate"""
        if self.total_input_frames > 0:
            drop_rate = ((self.total_input_frames - self.total_output_frames) / self.total_input_frames) * 100
            self.frame_drop_rate.labels(client_id=self.client_id).set(drop_rate)
            return drop_rate
        return 0

    def get_metrics_summary(self):
        """Get a summary of all metrics"""
        drop_rate = self.calculate_drop_rate()
        return {
            'input_frames': self.total_input_frames,
            'output_frames': self.total_output_frames,
            'drop_rate': f"{drop_rate:.2f}%"
        } 