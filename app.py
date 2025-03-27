# Add these at the very top of app.py, before any other imports
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow messages
import warnings
warnings.filterwarnings('ignore')

try:
    import torch
    import numpy as np
    from model import load_model
    import cv2
    import mediapipe as mp
    from flask import Flask, request, jsonify, Response, redirect, url_for, render_template, session
    from werkzeug.middleware.proxy_fix import ProxyFix
    from flask_limiter import Limiter
    from flask_limiter.util import get_remote_address
    from dotenv import load_dotenv
    import threading
    from datetime import datetime
    from database import ExerciseDatabase
    from flask_mail import Mail, Message
    from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
    import sqlite3
    from werkzeug.security import check_password_hash, generate_password_hash
    import logging
except ImportError as e:
    print(f"Error importing required packages: {str(e)}")
    print("Please make sure all required packages are installed")
    raise

# Load environment variables
load_dotenv()
print("Environment variables loaded:", {
    'EMAIL_USER': os.getenv('EMAIL_USER'),
    'SECRET_KEY': os.getenv('SECRET_KEY')
})

app = Flask(__name__, static_folder='static')
app.wsgi_app = ProxyFix(app.wsgi_app)

# Setup rate limiting
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"]
)

# Initialize Mediapipe Pose model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

# Configuration
MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10MB
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MODEL_PATH = os.getenv('MODEL_PATH', 'fitness_model.pth')
CLASSES = ['squats', 'benchpress', 'deadlift', 'pushup']

# Add these global variables
camera = None
output_frame = None
lock = threading.Lock()
rep_counter = 0
exercise_state = False  # False for starting position, True for rep position
last_prediction = None
last_time = datetime.now()

# Initialize database
db = ExerciseDatabase()
session_id = None

# Update the Flask app configuration
app.config.update(
    DEBUG=False,
    SECRET_KEY=os.getenv('SECRET_KEY'),  # Read from .env
    MAIL_SERVER='smtp.gmail.com',
    MAIL_PORT=587,
    MAIL_USE_TLS=True,
    MAIL_USERNAME=os.getenv('EMAIL_USER'),  # This will now be your new email
    MAIL_PASSWORD=os.getenv('EMAIL_PASS'),  # Your app password
    MAIL_DEFAULT_SENDER=os.getenv('EMAIL_USER')  # This will also use the new email
)

# Add debug print to check if environment variables are loaded
print("Email config:", {
    'MAIL_USERNAME': app.config['MAIL_USERNAME'],
    'MAIL_PASSWORD': 'HIDDEN',
    'MAIL_SERVER': app.config['MAIL_SERVER']
})

# Initialize extensions
mail = Mail(app)
# Add these at the top of your file with other imports
from flask_login import LoginManager

# Initialize LoginManager after creating the Flask app
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

@login_manager.user_loader
def load_user(user_id):
    conn = sqlite3.connect('fitness_data.db')
    c = conn.cursor()
    c.execute('SELECT * FROM users WHERE id = ?', (user_id,))
    user = c.fetchone()
    conn.close()
    
    if user:
        return User(user[0], user[1], user[2], user[3])
    return None

class User(UserMixin):
    def __init__(self, id, name, email, reg_number):
        self.id = id
        self.name = name
        self.email = email
        self.reg_number = reg_number

@login_manager.user_loader
def load_user(user_id):
    # Load user from database
    conn = sqlite3.connect('fitness_data.db')
    c = conn.cursor()
    c.execute('SELECT id, name, email, reg_number FROM users WHERE id = ?', (user_id,))
    user = c.fetchone()
    conn.close()
    if user:
        return User(*user)
    return None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to extract keypoints from an image
def extract_keypoints(image):
    results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if not results.pose_landmarks:
        return None
    landmarks = results.pose_landmarks.landmark
    keypoints = np.array([[lmk.x, lmk.y, lmk.z] for lmk in landmarks]).flatten()
    return keypoints

# Load the model with error handling
try:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(MODEL_PATH, input_size=99, num_classes=3).to(device)
    model.eval()
except Exception as e:
    print(f"Error loading model: {str(e)}")
    model = None

def calculate_angle(a, b, c):
    """Calculate angle between three points"""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle > 180.0:
        angle = 360-angle
    return angle

def validate_squat(landmarks):
    """Check if pose matches squat form"""
    # Get relevant landmarks
    hip = [landmarks[23].x, landmarks[23].y]
    knee = [landmarks[25].x, landmarks[25].y]
    ankle = [landmarks[27].x, landmarks[27].y]
    
    # Calculate knee angle
    knee_angle = calculate_angle(hip, knee, ankle)
    
    # Get back angle
    shoulder = [landmarks[11].x, landmarks[11].y]
    back_angle = calculate_angle(shoulder, hip, knee)
    
    # Simplified squat validation
    return knee_angle <= 170  # Only check if knees are bent

# Add these global variables at the top with other globals
# First, fix the global variables
rep_counters = {
    'squats': 0,
    'deadlift': 0,
    'benchpress': 0
}

# Remove the duplicate check_exercise_state function and keep only this version
def check_exercise_state(landmarks, prediction):
    global exercise_state, last_prediction, last_time, rep_counters
    
    current_time = datetime.now()
    time_diff = (current_time - last_time).total_seconds()
    
    if last_prediction != prediction:
        exercise_state = False
        last_prediction = prediction
        last_time = current_time
        return rep_counters[prediction]
    
    if prediction == 'squats':
        hip = [landmarks[23].x, landmarks[23].y]
        knee = [landmarks[25].x, landmarks[25].y]
        ankle = [landmarks[27].x, landmarks[27].y]
        knee_angle = calculate_angle(hip, knee, ankle)
        
        if not exercise_state and knee_angle < 110:
            exercise_state = True
            last_time = current_time
        elif exercise_state and knee_angle > 140 and time_diff > 0.5:  # Reduced time threshold
            exercise_state = False
            rep_counters[prediction] += 1
            last_time = current_time
            
    elif prediction == 'deadlift':
        shoulder = [landmarks[11].x, landmarks[11].y]
        hip = [landmarks[23].x, landmarks[23].y]
        knee = [landmarks[25].x, landmarks[25].y]
        hip_angle = calculate_angle(shoulder, hip, knee)
        
        if not exercise_state and hip_angle < 90:
            exercise_state = True
            last_time = current_time
        elif exercise_state and hip_angle > 150 and time_diff > 0.5:
            exercise_state = False
            rep_counters[prediction] += 1
            last_time = current_time
            
    elif prediction == 'benchpress':
        shoulder = [landmarks[11].x, landmarks[11].y]
        elbow = [landmarks[13].x, landmarks[13].y]
        wrist = [landmarks[15].x, landmarks[15].y]
        elbow_angle = calculate_angle(shoulder, elbow, wrist)
        
        if not exercise_state and elbow_angle < 100:
            exercise_state = True
            last_time = current_time
        elif exercise_state and elbow_angle > 150 and time_diff > 0.5:
            exercise_state = False
            rep_counters[prediction] += 1
            last_time = current_time
    
    return rep_counters[prediction]

# Update the generate_frames function
def generate_frames():
    global output_frame, lock, rep_counter, exercise_state
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    while True:
        success, frame = cap.read()
        if not success:
            break
        
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb_frame)
            
            if results.pose_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(
                    frame, 
                    results.pose_landmarks,
                    mp.solutions.pose.POSE_CONNECTIONS
                )
                
                landmarks = results.pose_landmarks.landmark
                keypoints = extract_keypoints(frame)
                
                if keypoints is not None and model is not None:
                    input_tensor = torch.tensor(keypoints, dtype=torch.float32).unsqueeze(0).to(device)
                    with torch.no_grad():
                        outputs = model(input_tensor)
                        _, predicted = torch.max(outputs, 1)
                        confidence = float(torch.softmax(outputs, 1)[0][predicted].item())
                        
                        if confidence > 0.3:
                            prediction = CLASSES[predicted.item()]
                            reps = check_exercise_state(landmarks, prediction)
                            
                            # Display current exercise
                            cv2.putText(frame, f"Exercise: {prediction}", 
                                      (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, 
                                      (0, 255, 0), 3)
                            
                            # Display all exercise counts
                            y_pos = 100
                            for ex, count in rep_counters.items():
                                color = (0, 255, 0) if ex == prediction else (200, 200, 0)
                                cv2.putText(frame, f"{ex}: {count} reps", 
                                          (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 1.2, 
                                          color, 2)
                                y_pos += 50
                            
                            # Display movement state
                            state_text = "DOWN" if exercise_state else "UP"
                            cv2.putText(frame, f"State: {state_text}", 
                                      (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 1.5, 
                                      (0, 255, 0), 3)
            
            _, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                   
        except Exception as e:
            print(f"Error processing frame: {str(e)}")
            continue
    
    cap.release()

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'model_loaded': model is not None})

@app.route('/predict', methods=['POST'])
@limiter.limit("10 per minute")
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 503

    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
        
    if not allowed_file(file.filename):
        return jsonify({'error': 'File type not allowed'}), 400
        
    # Check file size
    file.seek(0, os.SEEK_END)
    size = file.tell()
    file.seek(0)
    
    if size > MAX_IMAGE_SIZE:
        return jsonify({'error': 'File too large'}), 400

    try:
        image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), 1)
        if image is None:
            return jsonify({'error': 'Invalid image file'}), 400

        keypoints = extract_keypoints(image)
        if keypoints is None:
            return jsonify({'error': 'No keypoints detected'}), 400

        input_tensor = torch.tensor(keypoints, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(input_tensor)
            _, predicted = torch.max(outputs, 1)

        result = {
            'class': CLASSES[predicted.item()],
            'confidence': float(torch.softmax(outputs, 1)[0][predicted].item())
        }
        return jsonify(result)

    except Exception as e:
        return jsonify({'error': f'Processing error: {str(e)}'}), 500

@app.route('/camera')
@login_required
def camera():
    return render_template('camera.html')

# Update login route to use camera_demo instead
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        
        conn = sqlite3.connect('fitness_data.db')
        c = conn.cursor()
        c.execute('SELECT * FROM users WHERE email = ?', (email,))
        user = c.fetchone()
        conn.close()
        
        if user:
            stored_password = user[4]
            if check_password_hash(stored_password, password):
                user_obj = User(user[0], user[1], user[2], user[3])
                login_user(user_obj)
                return redirect(url_for('camera'))  # This will now work
        return 'Invalid email or password'
    
    return render_template('login.html')

@app.route('/camera_demo')
def camera_demo():
    return render_template('camera.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

# Modify the index route
@app.route('/')
def index():
    if current_user.is_authenticated:
        return redirect(url_for('camera'))
    return render_template('demo.html')  # Create this template to show login and demo options

# Update the register route to remove email verification
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        try:
            name = request.form['name']
            email = request.form['email']
            reg_number = request.form['reg_number']
            password = request.form['password']
            
            if db.create_user(name, email, reg_number, password):
                return redirect(url_for('login'))
            return "Email or registration number already exists"
        except Exception as e:
            print(f"Registration error: {str(e)}")
            return "Registration failed. Please try again."
    return render_template('register.html')

# Remove email-related routes
# Remove /verify_otp, /test_email routes

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)
        
