import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import csv
import math
import argparse
import sys
import time
import os

try:
    import psutil
except ImportError:
    print("Vui lòng chạy: pip install psutil")
    sys.exit(1)

class ObjectiveTrajectoryRecorder(Node):
    def __init__(self, topic_name, output_file):
        super().__init__('objective_recorder')
        self.output_file = output_file
        
        self.subscription = self.create_subscription(
            Twist,  
            topic_name,
            self.listener_callback,
            10)
        
        # Biến quỹ đạo
        self.pose_x = 0.0
        self.pose_y = 0.0
        self.pose_theta = 0.0
        self.last_time = self.get_clock().now()
        
        self.trajectory_data = [("x", "y")]

        # --- BỘ ĐO KHÁCH QUAN CẤP TIẾN TRÌNH (PROCESS LEVEL) ---
        self.current_process = psutil.Process(os.getpid())
        self.process_time_records = []
        self.memory_mb_records = []
        
        self.get_logger().info(f"Đang ghi dữ liệu từ '{topic_name}' vào '{self.output_file}'...")
        self.get_logger().info("Đang theo dõi chi phí tính toán ĐỘC LẬP của tiến trình này...")

    def listener_callback(self, msg):
        # 1. Bắt đầu đo thời gian CPU thực thi dành riêng cho tiến trình này
        t_start = time.process_time()

        # --- PHẦN TÍNH TOÁN CỦA THUẬT TOÁN (Ghi nhận quỹ đạo) ---
        current_time = self.get_clock().now()
        dt = (current_time - self.last_time).nanoseconds / 1e9
        
        v = msg.linear.x
        w = msg.angular.z
        
        self.pose_theta += w * dt
        self.pose_x += v * math.cos(self.pose_theta) * dt
        self.pose_y += v * math.sin(self.pose_theta) * dt
        
        self.trajectory_data.append((self.pose_x, self.pose_y))
        self.last_time = current_time
        # ---------------------------------------------------------

        # 2. Kết thúc đo thời gian CPU
        t_end = time.process_time()
        cpu_time_ms = (t_end - t_start) * 1000.0 # Đổi ra mili-giây
        
        # Lọc nhiễu (chỉ ghi nhận nếu thực sự có tốn time > 0)
        if cpu_time_ms > 0:
            self.process_time_records.append(cpu_time_ms)

        # 3. Lấy mẫu bộ nhớ RAM (RSS: Resident Set Size) tính bằng MB
        # Lấy định kỳ mỗi 10 step để tránh overhead hệ thống
        if len(self.trajectory_data) % 10 == 0:
            mem_mb = self.current_process.memory_info().rss / (1024 * 1024)
            self.memory_mb_records.append(mem_mb)

    def save_to_csv_and_report(self):
        print("\n" + "="*60)
        print("📊 BÁO CÁO CHI PHÍ TÍNH TOÁN KHÁCH QUAN (PROCESS-LEVEL)")
        print("="*60)
        
        if len(self.process_time_records) > 0:
            avg_time = sum(self.process_time_records) / len(self.process_time_records)
            max_time = max(self.process_time_records)
            min_time = min(self.process_time_records)
            
            print(f"⏱️ THỜI GIAN CPU (Process Time) trên mỗi vòng lặp:")
            print(f"   - Trung bình  : {avg_time:.4f} ms")
            print(f"   - Nhanh nhất  : {min_time:.4f} ms")
            print(f"   - Chậm nhất   : {max_time:.4f} ms")
            print(f"   -> AI thường có Min/Max gần nhau. Truyền thống thường có Min/Max chênh lệch lớn.")
        else:
            print("⚠️ Không đo được thời gian CPU (có thể thuật toán chạy quá nhanh hoặc chưa nhận data).")

        print("-" * 60)
        
        if len(self.memory_mb_records) > 0:
            avg_mem = sum(self.memory_mb_records) / len(self.memory_mb_records)
            max_mem = max(self.memory_mb_records)
            
            print(f"🧠 BỘ NHỚ RAM (Resident Set Size):")
            print(f"   - Dung lượng trung bình: {avg_mem:.2f} MB")
            print(f"   - Dung lượng đỉnh (Max): {max_mem:.2f} MB")
        else:
            print("⚠️ Không có dữ liệu RAM.")
            
        print("="*60)

        # Lưu file CSV
        if len(self.trajectory_data) > 1:
            try:
                with open(self.output_file, mode='w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerows(self.trajectory_data)
                print(f"\n[THÀNH CÔNG] Đã lưu quỹ đạo vào '{self.output_file}'")
            except Exception as e:
                print(f"\n[LỖI] Không thể lưu file CSV: {e}")

def main(args=None):
    rclpy.init(args=args)
    parser = argparse.ArgumentParser()
    parser.add_argument('--topic', type=str, default='/wheel_data')
    parser.add_argument('--output', type=str, required=True)
    
    parsed_args, _ = parser.parse_known_args(sys.argv[1:])
    recorder = ObjectiveTrajectoryRecorder(topic_name=parsed_args.topic, output_file=parsed_args.output)
    
    try:
        rclpy.spin(recorder)
    except KeyboardInterrupt:
        pass
    finally:
        recorder.save_to_csv_and_report()
        if rclpy.ok():
            recorder.destroy_node()
            rclpy.shutdown()

if __name__ == '__main__':
    main()
