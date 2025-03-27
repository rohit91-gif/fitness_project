from datetime import datetime
import sqlite3
import json
from werkzeug.security import generate_password_hash, check_password_hash
import random
import string
import os

class ExerciseDatabase:
    def __init__(self):
        self.db_path = os.getenv('DATABASE_URL', 'fitness_data.db')
        self.init_db()

    def init_db(self):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        # Add users table
        c.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT UNIQUE NOT NULL,
                reg_number TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                is_verified BOOLEAN DEFAULT FALSE
            )
        ''')
        
        # Add OTP table
        c.execute('''
            CREATE TABLE IF NOT EXISTS otps (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT NOT NULL,
                otp TEXT NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                expires_at DATETIME,
                is_used BOOLEAN DEFAULT FALSE
            )
        ''')
        
        # Modify workout_sessions to include user_id
        c.execute('''
            CREATE TABLE IF NOT EXISTS workout_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                start_time DATETIME,
                end_time DATETIME,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        ''')
        
        # Create tables for workout sessions and exercise data
        c.execute('''
            CREATE TABLE IF NOT EXISTS exercise_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER,
                timestamp DATETIME,
                exercise_type TEXT,
                form_correct BOOLEAN,
                feedback TEXT,
                angles TEXT,
                rep_count INTEGER,
                FOREIGN KEY (session_id) REFERENCES workout_sessions(id)
            )
        ''')
        
        conn.commit()
        conn.close()

    def create_user(self, name, email, reg_number, password):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        try:
            password_hash = generate_password_hash(password)
            c.execute('''
                INSERT INTO users (name, email, reg_number, password_hash)
                VALUES (?, ?, ?, ?)
            ''', (name, email, reg_number, password_hash))
            conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False
        finally:
            conn.close()

    def generate_otp(self, email):
        otp = ''.join(random.choices(string.digits, k=6))
        expires_at = datetime.now().replace(microsecond=0) + datetime.timedelta(minutes=10)
        
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('''
            INSERT INTO otps (email, otp, expires_at)
            VALUES (?, ?, ?)
        ''', (email, otp, expires_at))
        conn.commit()
        conn.close()
        return otp

    def verify_otp(self, email, otp):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('''
            SELECT id FROM otps 
            WHERE email = ? AND otp = ? AND expires_at > ? AND is_used = FALSE
            ORDER BY created_at DESC LIMIT 1
        ''', (email, otp, datetime.now().replace(microsecond=0)))
        result = c.fetchone()
        
        if result:
            otp_id = result[0]
            c.execute('UPDATE otps SET is_used = TRUE WHERE id = ?', (otp_id,))
            c.execute('UPDATE users SET is_verified = TRUE WHERE email = ?', (email,))
            conn.commit()
            conn.close()
            return True
        conn.close()
        return False

    def start_session(self, user_id):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('INSERT INTO workout_sessions (user_id, start_time) VALUES (?, ?)', 
                 (user_id, datetime.now()))
        session_id = c.lastrowid
        conn.commit()
        conn.close()
        return session_id

    def end_session(self, session_id):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('UPDATE workout_sessions SET end_time = ? WHERE id = ?', 
                 (datetime.now(), session_id))
        conn.commit()
        conn.close()

    def log_exercise(self, session_id, exercise_type, form_correct, feedback, angles, rep_count):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('''
            INSERT INTO exercise_data 
            (session_id, timestamp, exercise_type, form_correct, feedback, angles, rep_count)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (session_id, datetime.now(), exercise_type, form_correct, 
              json.dumps(feedback), json.dumps(angles), rep_count))
        conn.commit()
        conn.close()

    def get_session_summary(self, session_id):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        # Get session details
        c.execute('''
            SELECT start_time, end_time FROM workout_sessions 
            WHERE id = ?
        ''', (session_id,))
        session = c.fetchone()
        
        # Get exercise data
        c.execute('''
            SELECT exercise_type, form_correct, feedback, rep_count 
            FROM exercise_data 
            WHERE session_id = ?
            ORDER BY timestamp
        ''', (session_id,))
        exercises = c.fetchall()
        
        conn.close()
        
        return {
            'session': {
                'start_time': session[0],
                'end_time': session[1],
                'duration': str(datetime.fromisoformat(session[1]) - 
                               datetime.fromisoformat(session[0]))
            },
            'exercises': [{
                'type': ex[0],
                'form_correct': ex[1],
                'feedback': json.loads(ex[2]),
                'reps': ex[3]
            } for ex in exercises]
        }

    def get_user_by_email(self, email):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('''
            SELECT id, name, email, reg_number, password_hash 
            FROM users 
            WHERE email = ?
        ''', (email,))
        user = c.fetchone()
        conn.close()
        
        if user:
            return {
                'id': user[0],
                'name': user[1],
                'email': user[2],
                'reg_number': user[3],
                'password_hash': user[4]
            }
        return None

    def get_user_sessions(self, user_id):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        c.execute('''
            SELECT id, start_time, end_time 
            FROM workout_sessions 
            WHERE user_id = ? 
            ORDER BY start_time DESC
        ''', (user_id,))
        sessions = c.fetchall()
        
        result = []
        for session in sessions:
            session_id = session[0]
            c.execute('''
                SELECT exercise_type, form_correct, feedback, rep_count 
                FROM exercise_data 
                WHERE session_id = ?
                ORDER BY timestamp
            ''', (session_id,))
            exercises = c.fetchall()
            
            result.append({
                'id': session_id,
                'start_time': session[1],
                'end_time': session[2],
                'duration': str(datetime.fromisoformat(session[2]) - 
                              datetime.fromisoformat(session[1])) if session[2] else "Ongoing",
                'exercises': [{
                    'type': ex[0],
                    'form_correct': ex[1],
                    'feedback': json.loads(ex[2]),
                    'reps': ex[3]
                } for ex in exercises]
            })
        
        conn.close()
        return result 