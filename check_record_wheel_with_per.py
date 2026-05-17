import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import csv
import math
import argparse
import sys
import time

try:
    import psutil
except ImportError:
    print("[LỖI] Thư viện 'psutil' chưa được cài đặt.")
    print("Vui lòng mở terminal và chạy lệnh: pip install psutil")
    sys.exit(1)


class TrajectoryRecorder(Node):
    def __init__(self, topic_name, output_file):
        super().__init__('trajectory_recorder')
        self.output_file = output_file
        
        # Đăng ký lắng nghe topic. 
        # NẾU TYPE KHÁC TWIST, BẠN CẦN SỬA KIỂU DỮ LIỆU Ở ĐÂY!
        self.subscription = self.create_subscription(
            Twist,  
            topic_name,
            self.listener_callback,
            10)
        
        self.pose_x = 0.0
        self.pose_y = 0.0
        self.pose_theta = 0.0
        self.last_time = self.get_clock().now()
        
        self.trajectory_data = [("x", "y")]

        # --- BỘ BIẾN THEO DÕI CHI PHÍ TÍNH TOÁN (CPU/RAM) ---
        self.cpu_records = []
        self.ram_records = []
        self.start_time = time.time()
        
        # Tạo một Timer để lấy mẫu tài nguyên máy tính mỗi 1.0 giây
        self.profile_timer = self.create_timer(1.0, self.profile_system_callback)
        # Lần gọi đầu tiên để khởi tạo baseline cho CPU
        psutil.cpu_percent(interval=None)

        self.get_logger().info(f"Đang ghi quỹ đạo từ '{topic_name}' vào '{self.output_file}'...")
        self.get_logger().info("Đang theo dõi chi phí tính toán hệ thống (CPU & RAM)...")
        self.get_logger().info("Nhấn Ctrl+C để kết thúc và lưu báo cáo.")

    def profile_system_callback(self):
        """Lấy mẫu phần trăm sử dụng CPU và RAM của toàn bộ máy tính."""
        # interval=None giúp hàm không bị block (chờ đợi), phù hợp với môi trường ROS 2
        current_cpu = psutil.cpu_percent(interval=None)
        current_ram = psutil.virtual_memory().percent
        
        self.cpu_records.append(current_cpu)
        self.ram_records.append(current_ram)

    def listener_callback(self, msg):
        current_time = self.get_clock().now()
        dt = (current_time - self.last_time).nanoseconds / 1e9
        
        # Trích xuất dữ liệu (Cần sửa nếu msg không phải là Twist)
        v = msg.linear.x
        w = msg.angular.z
        
        self.pose_theta += w * dt
        self.pose_x += v * math.cos(self.pose_theta) * dt
        self.pose_y += v * math.sin(self.pose_theta) * dt
        
        self.trajectory_data.append((self.pose_x, self.pose_y))
        self.last_time = current_time

    def print_performance_report(self):
        """In ra Báo cáo chi phí tính toán của máy tính."""
        total_time = time.time() - self.start_time
        print("\n" + "="*50)
        print("📊 BÁO CÁO CHI PHÍ TÍNH TOÁN CỦA MÁY TÍNH")
        print("="*50)
        print(f"⏱️ Tổng thời gian chạy ghi nhận: {total_time:.2f} giây")

        if len(self.cpu_records) > 0:
            avg_cpu = sum(self.cpu_records) / len(self.cpu_records)
            max_cpu = max(self.cpu_records)
            avg_ram = sum(self.ram_records) / len(self.ram_records)
            max_ram = max(self.ram_records)

            print(f"💻 CPU Usage - Trung bình: {avg_cpu:.1f}% | Chạm đỉnh (Max): {max_cpu:.1f}%")
            print(f"🧠 RAM Usage - Trung bình: {avg_ram:.1f}% | Chạm đỉnh (Max): {max_ram:.1f}%")
        else:
            print("⚠️ Chưa thu thập đủ mẫu CPU/RAM (Thời gian chạy dưới 1 giây).")
        print("="*50)

    def save_to_csv(self):
        """Dùng print tiêu chuẩn thay vì logger của ROS 2 để tránh crash lúc tắt máy"""
        print("\n--- Đang xử lý dữ liệu đầu ra ---")
        
        # 1. In báo cáo tài nguyên hệ thống
        self.print_performance_report()
        
        # 2. Lưu file CSV
        if len(self.trajectory_data) > 1:
            try:
                with open(self.output_file, mode='w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerows(self.trajectory_data)
                print(f"[THÀNH CÔNG] Đã lưu {len(self.trajectory_data)-1} điểm tọa độ vào '{self.output_file}'\n")
            except Exception as e:
                print(f"[LỖI] Không thể lưu file CSV: {e}\n")
        else:
            print("[CẢNH BÁO] Không nhận được dữ liệu nào từ topic. Kiểm tra lại topic_name.\n")


def main(args=None):
    rclpy.init(args=args)
    
    parser = argparse.ArgumentParser(description='Ghi quỹ đạo robot và theo dõi tài nguyên ra file CSV.')
    parser.add_argument('--topic', type=str, default='/wheel_data', help='Tên topic lắng nghe')
    parser.add_argument('--output', type=str, required=True, help='Tên file CSV đầu ra (VD: algo1.csv)')
    
    parsed_args, _ = parser.parse_known_args(sys.argv[1:])
    
    recorder = TrajectoryRecorder(topic_name=parsed_args.topic, output_file=parsed_args.output)
    
    try:
        rclpy.spin(recorder)
    except KeyboardInterrupt:
        # Bỏ qua lỗi ngắt bàn phím để đi thẳng xuống finally
        pass
    finally:
        # Lưu file trước khi destroy node
        recorder.save_to_csv()
        
        # Rút gọn tắt máy an toàn
        if rclpy.ok():
            recorder.destroy_node()
            rclpy.shutdown()


if __name__ == '__main__':
    main()
