"""Behavior Cloning Inference Node.

Runs the trained PyTorch PolicyNet in real-time, processes LiDAR scans,
estimates trajectory, and measures computational cost (latency).
Outputs a trajectory plot and performance report upon shutdown.
"""

import argparse
import os
import sys
import threading
import time # Bổ sung thư viện đo thời gian

import numpy as np
import rclpy
import torch
import matplotlib.pyplot as plt
from geometry_msgs.msg import Twist
from models.model import PolicyNet
from rclpy.node import Node
from sensor_msgs.msg import LaserScan


def clamp_scan(ranges: list[float], max_range: float = 3.5) -> np.ndarray:
    """Handle NaN/Inf values and clip ranges."""
    arr = np.array(ranges, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=max_range, posinf=max_range, neginf=0.0)
    return np.clip(arr, 0.0, max_range)


class PolicyNode(Node):
    def __init__(
        self,
        model_path: str,
        downsample: int = 180,
        rate_hz: int = 20,
        max_range: float = 3.5,
    ):
        super().__init__("policy_node")
        self.downsample = downsample
        self.rate = rate_hz
        self.max_range = max_range

        self.scan: np.ndarray | None = None
        self.lock = threading.Lock()
        
        # Biến theo dõi quỹ đạo
        self.pose_x = 0.0
        self.pose_y = 0.0
        self.pose_theta = 0.0
        self.x_history: list[float] = [0.0]
        self.y_history: list[float] = [0.0]
        self.last_time = self.get_clock().now()

        # Biến theo dõi Chi phí tính toán (Độ trễ)
        self.latency_history: list[float] = []

        # UI Throttling
        self.step_counter = 0
        self.log_interval = max(1, int(self.rate / 2))

        # 1. Load PyTorch Model
        if not os.path.exists(model_path):
            self.get_logger().error(f"Model not found: {model_path}")
            sys.exit(1)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = PolicyNet(input_dim=self.downsample).to(self.device)

        try:
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.eval()
            self.get_logger().info(f"Model loaded on {self.device}")
        except Exception as e:
            self.get_logger().error(f"Failed to load weights: {e}")
            sys.exit(1)

        # 2. ROS 2 Communication
        self.scan_sub = self.create_subscription(LaserScan, "/scan", self.scan_cb, 10)
        self.pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.timer = self.create_timer(1.0 / float(self.rate), self.run_policy)

    def scan_cb(self, msg: LaserScan) -> None:
        with self.lock:
            self.scan = clamp_scan(msg.ranges, self.max_range)

    def downsample_scan(self, arr: np.ndarray) -> np.ndarray:
        if len(arr) < self.downsample:
            return np.pad(arr, (0, self.downsample - len(arr)), constant_values=self.max_range)
        idx = np.linspace(0, len(arr) - 1, self.downsample).astype(int)
        return arr[idx]

    def run_policy(self) -> None:
        """Timer loop for processing state and publishing actions."""
        # --- BẮT ĐẦU ĐO THỜI GIAN TÍNH TOÁN ---
        t_start = time.perf_counter()

        with self.lock:
            if self.scan is None:
                self.get_logger().warn("Waiting for /scan...", throttle_duration_sec=2.0)
                return
            current_scan = self.scan.copy()

        # Preprocessing
        scan_ds = self.downsample_scan(current_scan)
        min_dist = float(np.min(scan_ds))

        # Normalize and convert to tensor
        x_input = np.clip(scan_ds, 0.0, self.max_range) / self.max_range
        x_tensor = torch.tensor(x_input, dtype=torch.float32).unsqueeze(0).to(self.device)

        # Inference
        with torch.no_grad():
            out = self.model(x_tensor).squeeze(0).cpu().numpy()

        v = float(out[0])
        w = float(out[1])

        # Publish
        msg = Twist()
        msg.linear.x = v
        msg.angular.z = w
        self.pub.publish(msg)

        # Update Estimated Trajectory
        current_time = self.get_clock().now()
        dt = (current_time - self.last_time).nanoseconds / 1e9
        
        with self.lock:
            self.pose_theta += w * dt
            self.pose_x += v * np.cos(self.pose_theta) * dt
            self.pose_y += v * np.sin(self.pose_theta) * dt
            
            self.x_history.append(self.pose_x)
            self.y_history.append(self.pose_y)
            self.last_time = current_time

        # --- KẾT THÚC ĐO THỜI GIAN TÍNH TOÁN ---
        t_end = time.perf_counter()
        latency_ms = (t_end - t_start) * 1000.0  # Đổi ra mili-giây
        self.latency_history.append(latency_ms)

        # Throttled Logging
        self.step_counter += 1
        if self.step_counter % self.log_interval == 0:
            print(
                f"[RUNNING] min: {min_dist:.2f}m | "
                f"v: {v:+.2f} | w: {w:+.2f} | "
                f"Cost: {latency_ms:.1f} ms"
            )

    def print_performance_report(self) -> None:
        """In ra báo cáo chi phí tính toán tổng thể"""
        if not self.latency_history:
            return
            
        avg_latency = np.mean(self.latency_history)
        max_latency = np.max(self.latency_history)
        min_latency = np.min(self.latency_history)
        p99_latency = np.percentile(self.latency_history, 99) # Độ trễ ở phân vị 99%
        avg_fps = 1000.0 / avg_latency if avg_latency > 0 else 0.0

        print("\n" + "="*45)
        print("📊 BÁO CÁO CHI PHÍ TÍNH TOÁN (PERFORMANCE)")
        print("="*45)
        print(f"Tổng số chu kỳ đã chạy : {len(self.latency_history)} steps")
        print(f"Tốc độ xử lý trung bình: {avg_fps:.2f} Hz (FPS)")
        print(f"Độ trễ trung bình      : {avg_latency:.2f} ms")
        print(f"Độ trễ thấp nhất (Min) : {min_latency:.2f} ms")
        print(f"Độ trễ cao nhất (Max)  : {max_latency:.2f} ms")
        print(f"Độ trễ 99% (P99)       : {p99_latency:.2f} ms (99% số vòng lặp nhanh hơn mức này)")
        print("="*45 + "\n")

    def destroy_node(self) -> None:
        # Dừng robot
        self.pub.publish(Twist())
        self.get_logger().info("Zero velocity sent.")
        
        # In báo cáo hiệu năng tính toán
        self.print_performance_report()
        
        # Vẽ biểu đồ quỹ đạo
        with self.lock:
            if len(self.x_history) > 1:
                plt.figure(figsize=(8, 8))
                plt.plot(self.x_history, self.y_history, label="Estimated Trajectory", color='blue', linewidth=2)
                plt.scatter(self.x_history[0], self.y_history[0], color='green', label="Start", zorder=5)
                plt.scatter(self.x_history[-1], self.y_history[-1], color='red', label="End", zorder=5)
                
                plt.title("Robot Estimated Position Trajectory")
                plt.xlabel("X Position (m)")
                plt.ylabel("Y Position (m)")
                plt.legend()
                plt.grid(True)
                plt.axis("equal")
                
                plot_path = "robot_trajectory_estimated.png"
                plt.savefig(plot_path, bbox_inches='tight')
                self.get_logger().info(f"Trajectory plot saved to: {plot_path}")

        super().destroy_node()


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--downsample", type=int, default=180)
    parser.add_argument("--rate", type=int, default=20)
    parser.add_argument("--max_range", type=float, default=3.5)

    parsed, _ = parser.parse_known_args()

    node = PolicyNode(
        model_path=parsed.model,
        downsample=parsed.downsample,
        rate_hz=parsed.rate,
        max_range=parsed.max_range,
    )

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
