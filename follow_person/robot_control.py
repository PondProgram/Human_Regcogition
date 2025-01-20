import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32MultiArray

class RobotMover(Node):
    def __init__(self):
        super().__init__('robot_mover_node')
        self.subscription = self.create_subscription(
            Float32MultiArray,
            'robot_control',
            self.listener_callback,
            10)
        self.publisher_ = self.create_publisher(Twist, 'cmd_vel', 10)
        self.threshold = 80.00  

    def listener_callback(self, msg):
        x_deviation, y_deviation = msg.data
        self.move_robot(x_deviation, y_deviation)
        
    def move_robot(self, x_deviation, y_deviation):
        max_linear_speed = 0.5  
        max_angular_speed = 0.5  

        twist = Twist()

        if x_deviation is not None and y_deviation is not None:
            # ตรวจสอบระยะห่าง
            if y_deviation < 50:
                # ถอยหลัง
                linear_speed = -(max_linear_speed * (y_deviation / (y_deviation + self.threshold)))
                twist.linear.x = min(linear_speed, max_linear_speed)
                print(f"......................Move Backward with speed: {twist.linear.x:.2f}......................")
            elif y_deviation > 100:
                # เดินหน้า
                linear_speed = max_linear_speed * (y_deviation / (y_deviation + self.threshold))
                twist.linear.x = min(linear_speed, max_linear_speed)
                print(f"......................Move Forward with speed: {twist.linear.x:.2f}......................")
            else:
                # หยุด
                twist.linear.x = 0.0
                print("......................Robot Stop......................")

            # การหมุน (Angular Movement)
            if abs(x_deviation) > self.threshold:
                angular_speed = max_angular_speed * (x_deviation / (abs(x_deviation) + self.threshold))
                twist.angular.z = max(min(angular_speed, max_angular_speed), -max_angular_speed)
                if x_deviation > 0:
                    print(f"......................Turn Left with speed: {twist.angular.z:.2f}......................")
                else:
                    print(f"......................Turn Right with speed: {twist.angular.z:.2f}......................")
            else:
                twist.angular.z = 0.0

        self.publisher_.publish(twist)


def main(args=None):
    rclpy.init(args=args)
    robot_mover = RobotMover()
    try:
        rclpy.spin(robot_mover)
    except KeyboardInterrupt:
        pass
    finally:
        robot_mover.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
