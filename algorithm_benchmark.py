import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import psutil
import csv
import math
import argparse
import sys
import time
import os
import numpy as np

class BenchmarkNode(Node):
    def __init__(self, algo_name, process_keyword, wheel_topic, cmd_vel_topic):
        super().__init__('algorithm_benchmark')
        self.algo_name = algo_name
        self.output_csv = f"{algo_name}_benchmark_data.csv"
        
        # 1. Tìm tiến trình (Process) của thuật toán để đo RAM/CPU
        self.target_process = self.find_process(process_keyword)
        if not self.target_process:
            self.get_logger().error(f"Không tìm thấy tiến trình nào chứa từ khóa '{process_keyword}'.")
            self.get_logger().error("Vui lòng đảm bảo thuật toán đang chạy trước khi bật file này!")
            sys.exit(1)
            
        self.get_logger().info(f"[ĐÃ TÌM THẤY] Theo dõi tiến trình: {self.target_process.name()} (PID: {self.target_process.pid})")

        # 2. Đăng ký Topics
        self.sub_wheel = self.create_subscription(Twist, wheel_topic, self.wheel_callback, 10)
        self.sub_cmd = self.create_subscription(Twist, cmd_vel_topic, self.cmd_callback, 10)

        # 3. Biến lưu trữ dữ liệu
        # Quỹ đạo
        self.pose_x, self.pose_y, self.pose_theta = 0.0, 0.0, 0.0
        self.last_wheel_time = self.get_clock().now()
        
        # Độ trễ (Jitter / Cycle Time)
        self.last_cmd_time = None
        self.cycle_times_ms = []
        
        # Tài nguyên hệ thống (RAM, CPU)
        self.ram_records_mb = []
        self.cpu_records_percent = []
        
        # Dữ liệu gộp để xuất CSV: [Timestamp, X, Y, RAM_MB, CPU_Percent, Cycle_Time_ms]
        self.full_log = [("Time_s", "X", "Y", "RAM_MB", "CPU_Percent", "Cycle_Time_ms")]
        self.start_time = time.time()

        # 4. Timer đo tài nguyên (10Hz)
        self.profile_timer = self.create_timer(0.1, self.profile_system)
        self.get_logger().info(f"Đang ghi nhận Benchmarking cho thuật toán: {self.algo_name}...")

    def find_process(self, keyword):
        """Tìm PID của tiến trình dựa trên từ khóa truyền vào (vd: 'controller_server' hoặc 'hg_dagger')"""
        for p in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                # Kiểm tra trong command line arguments
                if p.info['cmdline'] and any(keyword in arg for arg in p.info['cmdline']):
                    return p
                # Kiểm tra trong tên tiến trình
                if p.info['name'] and keyword in p.info['name']:
                    return p
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return None

    def profile_system(self):
        """Lấy mẫu RAM và CPU của riêng tiến trình đích (10 lần/giây)"""
        try:
            # RSS (Resident Set Size): Lượng RAM vật lý thực tế đang dùng
            ram_mb = self.target_process.memory_info().rss / (1024 * 1024)
            # CPU phần trăm của riêng process này
            cpu_percent = self.target_process.cpu_percent(interval=None)
            
            self.ram_records_mb.append(ram_mb)
            self.cpu_records_percent.append(cpu_percent)
            
            # Ghi log thời gian thực vào bảng dữ liệu
            elapsed = time.time() - self.start_time
            last_cycle = self.cycle_times_ms[-1] if self.cycle_times_ms else 0.0
            self.full_log.append((elapsed, self.pose_x, self.pose_y, ram_mb, cpu_percent, last_cycle))
            
        except psutil.NoSuchProcess:
            self.get_logger().warn("Tiến trình mục tiêu đã bị đóng!")
            self.profile_timer.cancel()

    def wheel_callback(self, msg):
        """Tính toán quỹ đạo Dead Reckoning"""
        current_time = self.get_clock().now()
        dt = (current_time - self.last_wheel_time).nanoseconds / 1e9
        v, w = msg.linear.x, msg.angular.z
        
        self.pose_theta += w * dt
        self.pose_x += v * math.cos(self.pose_theta) * dt
        self.pose_y += v * math.sin(self.pose_theta) * dt
        self.last_wheel_time = current_time

    def cmd_callback(self, msg):
        """Đo khoảng cách thời gian giữa 2 lệnh cmd_vel liên tiếp để đánh giá độ trễ xử lý"""
        current_time = self.get_clock().now()
        if self.last_cmd_time is not None:
            dt_ms = (current_time - self.last_cmd_time).nanoseconds / 1e6
            # Chỉ ghi nhận các chu kỳ hợp lệ để tránh nhiễu lúc mới bật
            if dt_ms > 0 and dt_ms < 1000: 
                self.cycle_times_ms.append(dt_ms)
        self.last_cmd_time = current_time

    def print_and_save_report(self):
        print("\n" + "="*65)
        print(f"🏆 BÁO CÁO BENCHMARK TỔNG THỂ: {self.algo_name.upper()}")
        print("="*65)

        # 1. Báo cáo Độ trễ & Ổn định (Latency / Cycle Time)
        if self.cycle_times_ms:
            avg_cycle = np.mean(self.cycle_times_ms)
            min_cycle = np.min(self.cycle_times_ms)
            max_cycle = np.max(self.cycle_times_ms)
            var_cycle = np.var(self.cycle_times_ms)
            print(f"⏱️ CHU KỲ XUẤT LỆNH (Cycle Time - Proxy for Latency):")
            print(f"   - Tần số trung bình: {1000.0/avg_cycle:.1f} Hz ({avg_cycle:.1f} ms/lệnh)")
            print(f"   - Độ trễ Min       : {min_cycle:.1f} ms")
            print(f"   - Độ trễ Max (Spike) : {max_cycle:.1f} ms (Sự kiện chậm nhất)")
            print(f"   - Phương sai (Var) : {var_cycle:.2f} (Càng nhỏ càng ổn định)")
        else:
            print("⚠️ Không đo được chu kỳ xuất lệnh (Chưa nhận được /cmd_vel).")
            
        print("-" * 65)

        # 2. Báo cáo RAM & CPU
        if self.ram_records_mb:
            print(f"🧠 BỘ NHỚ RAM DÙNG RIÊNG (Resident Set Size):")
            print(f"   - Trung bình       : {np.mean(self.ram_records_mb):.1f} MB")
            print(f"   - Cao nhất (Peak)  : {np.max(self.ram_records_mb):.1f} MB")
            print(f"\n💻 CPU SỬ DỤNG (Process CPU):")
            print(f"   - Trung bình       : {np.mean(self.cpu_records_percent):.1f} %")
            print(f"   - Cao nhất (Peak)  : {np.max(self.cpu_records_percent):.1f} %")
        else:
            print("⚠️ Không đo được tài nguyên CPU/RAM.")
            
        print("="*65)

        # Lưu CSV
        if len(self.full_log) > 1:
            try:
                with open(self.output_csv, mode='w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerows(self.full_log)
                print(f"[THÀNH CÔNG] Dữ liệu chi tiết đã được xuất ra: {self.output_csv}")
                print(f"Bạn có thể dùng file này để vẽ đồ thị RAM, CPU, Cycle Time theo trục thời gian (Time_s).")
            except Exception as e:
                print(f"[LỖI] Không thể lưu file CSV: {e}")

def main(args=None):
    rclpy.init(args=args)
    parser = argparse.ArgumentParser(description='Công cụ Benchmark Thuật toán độc lập.')
    parser.add_argument('--algo_name', type=str, required=True, help='Tên thuật toán (vd: hg_dagger, dwb)')
    parser.add_argument('--keyword', type=str, required=True, help='Từ khóa tìm tiến trình (vd: controller_server)')
    parser.add_argument('--wheel_topic', type=str, default='/wheel_data')
    parser.add_argument('--cmd_topic', type=str, default='/cmd_vel')
    
    parsed_args, _ = parser.parse_known_args(sys.argv[1:])
    
    node = BenchmarkNode(
        algo_name=parsed_args.algo_name, 
        process_keyword=parsed_args.keyword,
        wheel_topic=parsed_args.wheel_topic,
        cmd_vel_topic=parsed_args.cmd_topic
    )
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.print_and_save_report()
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()

if __name__ == '__main__':
    main()
