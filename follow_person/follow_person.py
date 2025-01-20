import rclpy
from rclpy.node import Node
import cv2
import time
import sys
import cvzone
import numpy as np
sys.path.append('/home/thanawat/amr_ws/src/follow_person/follow_person')
from custom_face_recognition import FaceRecognition
from human_detection import HumanDetection
from std_msgs.msg import Float32MultiArray
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

class PersonTracker:
    def __init__(self, person_id, name, bbox):
        self.person_id = person_id
        self.name = name
        self.bbox = bbox
        self.last_detection_time = time.time()
        self.is_tracked = True
        self.kalman = cv2.KalmanFilter(4, 2)  # 4 state variables (x, y, dx, dy), 2 measurement variables (x, y)
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        self.kalman.processNoiseCov = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32) * 0.03
        
        # Initialize Kalman state with first detection
        self.kalman.statePre = np.array([[bbox[0]], [bbox[1]], [0], [0]], np.float32)
        self.kalman.statePost = np.array([[bbox[0]], [bbox[1]], [0], [0]], np.float32)

class ImageProcessor:
    def __init__(self):
        self.bridge = CvBridge()
        
    def process_frame(self, frame):
     
        # Resize frame to 240x160
        resized_frame = cv2.resize(frame, (240, 160))
        
        # Convert OpenCV image to ROS Image message
        ros_image = self.bridge.cv2_to_imgmsg(resized_frame, encoding="bgr8")
        
        return ros_image

class RobotControl:
    def __init__(self):
        # self.margin = 50  # ระยะห่างจากจุดกึ่งกลางเฟรม (pixel)
        pass
    
    def process_frame(self, frame, boxes, labels):

        height, width = frame.shape[:2]
        
        # กรองเอาเฉพาะ box ที่เป็น "Person" หรือ "Tracking"
        person_boxes = [box for box, label in zip(boxes, labels) if "Person" in label]
        tracking_boxes = [box for box, label in zip(boxes, labels) if "Tracking" in label]
        
        # เงื่อนไขที่ให้ทำงานต่อ
        if (len(person_boxes) == 1 and len(tracking_boxes) == 0):
            valid_boxes = person_boxes
        elif (len(tracking_boxes) == 1 and len(person_boxes) == 0):
            valid_boxes = tracking_boxes
        else:
            return None
            
        # คำนวณค่าต่างๆ จาก box ที่ถูกต้อง
        x, y, w, h = valid_boxes[0]
        
        # หาจุดกึ่งกลาง box
        center_x_obj = x + w // 2
        
        # คำนวณ deviation
        x_deviation = round((width // 2) - center_x_obj, 3)
        y_deviation = round(y, 3)  # วัดจากด้านบนของเฟรม
        
        return x_deviation, y_deviation


class CombinedSystem:
    def __init__(self, data_directory):
        self.face_recognition = FaceRecognition(data_directory)
        self.human_detection = HumanDetection(data_directory)
        self.robot_control = RobotControl()  
        self.tracked_persons = {}
        self.next_id = 1
        self.recognition_timeout = 5.0
        self.target_name = "For"
        
    def update_tracker(self, tracker, bbox):
        # Predict next position
        prediction = tracker.kalman.predict()
        
        # Update with new measurement
        measurement = np.array([[bbox[0]], [bbox[1]]], np.float32)
        tracker.kalman.correct(measurement)
        
        # Update bbox with Kalman estimate
        estimated_pos = tracker.kalman.statePost
        tracker.bbox = [int(estimated_pos[0][0]), int(estimated_pos[1][0]), bbox[2], bbox[3]]
        tracker.last_detection_time = time.time()
        tracker.is_tracked = True

    def process_frame(self, frame):
        frame = self.face_recognition.preprocess_image(frame)
        current_time = time.time()
        
        # เก็บ boxes และ labels สำหรับส่งให้ robot_control
        detected_boxes = []
        detected_labels = []
        
        # Human detection
        classIds, confs, bbox = self.human_detection.detect_humans(frame)
        
        # Update tracking status for existing tracked persons
        for person_id, tracker in list(self.tracked_persons.items()):
            if current_time - tracker.last_detection_time > self.recognition_timeout:
                if tracker.name == self.target_name:
                    tracker.is_tracked = False
                else:
                    del self.tracked_persons[person_id]
        
        tracked_target = None
        for pid, tracker in self.tracked_persons.items():
            if tracker.name == self.target_name:
                tracked_target = tracker
                break
        
        if len(classIds) > 0:
            for classId, conf, box in zip(classIds.flatten(), confs.flatten(), bbox):
                if classId == 1:  # Person class
                    x, y, w, h = box
                    margin = 40
                    x = max(0, x - margin)
                    y = max(0, y - margin)
                    w = min(frame.shape[1] - x, w + 2*margin)
                    h = min(frame.shape[0] - y, h + 2*margin)
                    
                    person_region = frame[y:y+h, x:x+w]
                    if person_region.size == 0:
                        continue
                    
                    # Face recognition on person region
                    features, faces = self.face_recognition.recognize_face(person_region)
                    
                    # Default state
                    label = "Human"
                    box_color = (255, 255, 0)  # Yellow for Human
                    
                    target_detected = False
                    
                    if faces is not None and len(faces) > 0:
                        for idx, (face, feature) in enumerate(zip(faces, features)):
                            result, user = self.face_recognition.match(feature)
                            
                            if result:
                                id_name, score = user
                                
                                # Handle target person
                                if id_name == self.target_name:
                                    target_detected = True
                                    
                                    if tracked_target is None:
                                        # Create new tracker for target
                                        tracked_id = self.next_id
                                        self.next_id += 1
                                        self.tracked_persons[tracked_id] = PersonTracker(tracked_id, self.target_name, box)
                                        tracked_target = self.tracked_persons[tracked_id]
                                    
                                    # Update existing tracker
                                    self.update_tracker(tracked_target, box)
                                    
                                    label = f"Person (ID: {tracked_target.person_id})"
                                    box_color = (0, 255, 0)  # Green for Person
                                
                                # Draw face detection box
                                face_box = list(map(int, face[:4]))
                                face_box[0] += x
                                face_box[1] += y

                                x, y, w, h = box
                                center_x_obj = x + w // 2
                                center_y_obj = y + h // 2
                                cv2.rectangle(frame, 
                                            (face_box[0], face_box[1]),
                                            (face_box[0] + face_box[2], face_box[1] + face_box[3]),
                                            box_color, 2)
                                cv2.circle(frame, (center_x_obj, center_y_obj), 5, (255, 0, 0), -1)
                                
                                # Add recognition text
                                text = f"{id_name} ({score:.2f})"
                                cv2.putText(frame, text,
                                        (face_box[0], face_box[1] - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)
                    
                    # If we're tracking target and detect a person near the predicted position
                    if tracked_target and not target_detected:
                        predicted_box = tracked_target.bbox
                        # Calculate center points
                        pred_center = (predicted_box[0] + predicted_box[2]//2, predicted_box[1] + predicted_box[3]//2)
                        curr_center = (box[0] + box[2]//2, box[1] + box[3]//2)
                        
                        # Calculate distance between predicted and current position
                        distance = np.sqrt((pred_center[0] - curr_center[0])**2 + (pred_center[1] - curr_center[1])**2)
                        
                        # If person is close to predicted position, update tracker
                        if distance < 100:  # Threshold can be adjusted
                            self.update_tracker(tracked_target, box)
                            label = f"Tracking {tracked_target.name} (ID: {tracked_target.person_id})"
                            box_color = (0, 165, 255)  # Orange for tracking without face detection
                    
                    # Draw bounding box and label
                    cvzone.cornerRect(frame, box, colorC=box_color)
                    x, y, w, h = box
                    center_x_obj = x + w // 2
                    center_y_obj = y + h // 2
                    cv2.circle(frame, (center_x_obj, center_y_obj), 5, (255, 0, 0), -1)
                    cv2.putText(frame, label,
                            (box[0], box[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, box_color, 2)
                    
                    # เพิ่ม box และ label เข้า list
                    detected_boxes.append(box)
                    detected_labels.append(label)
        
        # คำนวณค่าควบคุมหลังจบการ detect
        control_values = self.robot_control.process_frame(frame, detected_boxes, detected_labels)
        
        return frame, control_values
    
    def __del__(self):
        cv2.destroyAllWindows()

class FaceRecognitionNode(Node):
    def __init__(self):
        super().__init__('face_recognition_node')
        
        # เพิ่ม publisher สำหรับส่งค่าควบคุม
        self.control_publisher = self.create_publisher(
            Float32MultiArray,
            'robot_control',
            10
        )
        
        self.image_publisher = self.create_publisher(
            Image,
            'camera_feed',
            10
        )

        # Initialize the combined system
        self.data_directory = '/home/thanawat/amr_ws/src/follow_person/data' 
        self.system = CombinedSystem(self.data_directory)
        self.image_processor = ImageProcessor()
        # Initialize camera
        self.capture = cv2.VideoCapture(2)
        if not self.capture.isOpened():
            self.get_logger().error("Error: Could not open camera")
            sys.exit()
        
        # Create timer for processing frames
        self.timer = self.create_timer(0.033, self.process_frame) 
        
        # For FPS calculation
        self.last_time = time.time()
        
        self.get_logger().info('Face Recognition Node has been started')

    def process_frame(self):
        try:
            # Read frame from camera
            ret, current_frame = self.capture.read()
            if not ret:
                self.get_logger().error("Error: Could not read frame")
                return
            
            # รับทั้งเฟรมและค่าควบคุมจาก process_frame
            processed_frame, control_values = self.system.process_frame(current_frame)
            
            # ส่งค่าควบคุมถ้ามีการคำนวณได้
            if control_values is not None:
                msg = Float32MultiArray()
                msg.data = [float(control_values[0]), float(control_values[1])]
                self.control_publisher.publish(msg)
            
            # Calculate FPS
            current_time = time.time()
            fps = 1 / (current_time - self.last_time)
            self.last_time = current_time
            
            self.margin  = 80
            height, width, _ = processed_frame.shape
            center_x = width // 2
            center_y = height // 2
            cv2.line(processed_frame, (center_x - self.margin , 0), 
                     (center_x - self.margin, height), (0, 0, 255), 2)
            cv2.line(processed_frame, (center_x + self.margin , 0), 
                     (center_x + self.margin, height), (0, 0, 255), 2)
            cv2.line(processed_frame, (center_x - 100, 100), (center_x + 100, 100), (0, 255, 255), 2)
            cv2.line(processed_frame, (center_x - 100, 50), (center_x + 100, 50), (0, 255, 255), 2)


            
            # Add FPS to the frame
            cv2.putText(processed_frame, f'FPS: {int(fps)}', (20, 70),
                        cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 2)
            
            # Display the processed frame
            ros_image = self.image_processor.process_frame(processed_frame)
            self.image_publisher.publish(ros_image)
            cv2.imshow("Face Recognition System", processed_frame)
            
            # Check for 'q' key to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.destroy_node()
                rclpy.shutdown()
                
        except Exception as e:
            self.get_logger().error(f'Error processing frame: {str(e)}')

    def __del__(self):
        self.capture.release()
        cv2.destroyAllWindows()

def main(args=None):
    rclpy.init(args=args)
    node = FaceRecognitionNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.capture.release()
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()