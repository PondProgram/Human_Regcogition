import cv2
import time
import sys
import cvzone
from face_recognition import FaceRecognition
from human_detection import HumanDetection

class CombinedSystem:
    def __init__(self, data_directory):
        self.face_recognition = FaceRecognition(data_directory)
        self.human_detection = HumanDetection(data_directory)
    
    def process_frame(self, frame):
        frame = self.face_recognition.preprocess_image(frame)
        
        # Human detection
        classIds, confs, bbox = self.human_detection.detect_humans(frame)
        
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
                    
                    # Default label is 'Human'
                    label = "Human"
                    box_color = (255, 255, 0)  # Yellow for Human
                    
                    if faces is not None and len(faces) > 0:
                        recognized_person = False
                        
                        for idx, (face, feature) in enumerate(zip(faces, features)):
                            result, user = self.face_recognition.match(feature)
                            
                            # Adjust face coordinates
                            face_box = list(map(int, face[:4]))
                            face_box[0] += x
                            face_box[1] += y
                            
                            if result:
                                label = "Person"
                                box_color = (0, 255, 0)  # Green for Person
                                recognized_person = True
                            
                            # Draw face detection and recognition results
                            face_color = (0, 255, 0) if result else (0, 0, 255)
                            cv2.rectangle(frame, 
                                        (face_box[0], face_box[1]),
                                        (face_box[0] + face_box[2], face_box[1] + face_box[3]),
                                        face_color, 2)
                            
                            # Add recognition result
                            id_name, score = user if result else (f"Unknown_{idx}", 0.0)
                            text = f"{id_name} ({score:.2f})"
                            cv2.putText(frame, text,
                                      (face_box[0], face_box[1] - 10),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, face_color, 2)
                        
                        cvzone.cornerRect(frame, box, colorC=box_color)
                        cv2.putText(frame, label,
                                  (box[0], box[1] - 10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.8, box_color, 2)
                    else:
                        cvzone.cornerRect(frame, box, colorC=box_color)
                        cv2.putText(frame, label,
                                  (box[0], box[1] - 10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.8, box_color, 2)
        
        return frame

def main():
    data_directory = 'C:/Users/HP/Desktop/Model_Human_Recognition/data'  
    system = CombinedSystem(data_directory)
    
    capture = cv2.VideoCapture(0)
    if not capture.isOpened():
        print("Error: Could not open camera")
        sys.exit()
    
    print("Press 'q' to quit")
    
    while True:
        start_time = time.time()
        ret, frame = capture.read()
        if not ret:
            print("Error: Could not read frame")
            break
            
        processed_frame = system.process_frame(frame)
        
        # Calculate and display FPS
        fps = 1 / (time.time() - start_time)
        cv2.putText(processed_frame, f'FPS: {int(fps)}', (20, 70),
                    cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)
        
        cv2.imshow("Face Recognition System", processed_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    capture.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()