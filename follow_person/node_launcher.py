import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import subprocess
import shlex
import signal
import psutil

class CommandHandlerNode(Node):
    def __init__(self):
        super().__init__('command_handler_node')
        self.subscription = self.create_subscription(
            String,
            'node_control_topic',
            self.command_callback,
            10
        )
        self.process = None
        self.current_command = None
        self.get_logger().info('Command Handler Node is ready.')

    def command_callback(self, msg):
        command = msg.data.strip()
        self.get_logger().info(f'Received command: {command}')
        
        if command.startswith('ros2 launch') or command.startswith('ros2 run'):
            if self.process is not None:
                self.get_logger().warn('A process is already running. Stop it first.')
            else:
                self.current_command = command
                self.start_process(command)
        elif command.startswith('stop '):
            stop_command = command.replace('stop ', '', 1).strip()
            if self.process is not None and stop_command == self.current_command:
                self.stop_process()
            else:
                self.get_logger().warn('No matching process to stop.')
        else:
            self.get_logger().warn('Invalid command received.')

    def start_process(self, command):
        try:
            args = shlex.split(command)
            self.process = subprocess.Popen(args, preexec_fn=lambda: signal.signal(signal.SIGINT, signal.SIG_IGN))
            self.get_logger().info(f'Started process: {command}')
        except Exception as e:
            self.get_logger().error(f'Failed to start process: {e}')
            self.current_command = None

    def stop_process(self):
        if self.process is not None:
            try:
                # Get the process group
                parent = psutil.Process(self.process.pid)
                # Kill all child processes
                for child in parent.children(recursive=True):
                    child.terminate()
                # Kill parent process
                parent.terminate()
                
                # Wait and force kill if needed
                gone, alive = psutil.wait_procs(parent.children(), timeout=3)
                if alive:
                    for p in alive:
                        p.kill()
                if parent.is_running():
                    parent.kill()
                    
                self.process = None
                self.current_command = None
                self.get_logger().info('Process stopped successfully.')
                
            except (psutil.NoSuchProcess, psutil.AccessDenied, Exception) as e:
                self.get_logger().error(f'Failed to stop process: {e}')
                self.process = None
                self.current_command = None
        else:
            self.get_logger().warn('No process is running.')

def main(args=None):
    rclpy.init(args=args)
    node = CommandHandlerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down node...')
    finally:
        if node.process is not None:
            node.stop_process()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()