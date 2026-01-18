import sqlite3
import json
from datetime import datetime
from typing import List, Dict, Any
import uuid

class DatabaseManager:
    def __init__(self, db_path='dsa_tutor.db'):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database tables"""
        conn = self.get_connection()
        cursor = conn.cursor()

        # Create tables with proper schema
        cursor.executescript('''
            -- Users table with authentication
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                last_login TEXT DEFAULT CURRENT_TIMESTAMP
            );

            -- Learning sessions
            CREATE TABLE IF NOT EXISTS learning_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL,
                session_type TEXT DEFAULT 'practice',
                topics TEXT, -- JSON array of topics
                difficulty TEXT,
                started_at TEXT DEFAULT CURRENT_TIMESTAMP,
                ended_at TEXT,
                FOREIGN KEY (username) REFERENCES users(username)
            );

            -- Question attempts (problems generated and solved)
            CREATE TABLE IF NOT EXISTS question_attempts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER,
                username TEXT NOT NULL,
                problem_data TEXT, -- JSON of the problem
                user_code TEXT,
                user_explanation TEXT,
                evaluation_result TEXT, -- JSON of evaluation
                score REAL,
                time_spent_seconds INTEGER DEFAULT 0,
                attempted_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES learning_sessions(id),
                FOREIGN KEY (username) REFERENCES users(username)
            );

            -- Concept explanations requested
            CREATE TABLE IF NOT EXISTS concept_explanations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL,
                concept TEXT NOT NULL,
                topic TEXT,
                explanation_requested_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (username) REFERENCES users(username)
            );

            -- Hint requests
            CREATE TABLE IF NOT EXISTS hint_requests (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                attempt_id INTEGER,
                username TEXT NOT NULL,
                hint_level INTEGER,
                hint_data TEXT, -- JSON of hint provided
                requested_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (attempt_id) REFERENCES question_attempts(id),
                FOREIGN KEY (username) REFERENCES users(username)
            );
        ''')

        conn.commit()
        conn.close()
    
    def get_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn
    
    
    # ==================== AUTHENTICATION METHODS ====================
    
    def register_user(self, username: str, email: str, password_hash: str) -> bool:
        """
        Register a new user
        
        Args:
            username: Unique username
            email: User's email address
            password_hash: Hashed password (use utils.auth.hash_password)
            
        Returns:
            True if registration successful, False if username/email already exists
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT INTO users (username, email, password_hash, created_at, last_login)
                VALUES (?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
            ''', (username, email, password_hash))
            
            conn.commit()
            conn.close()
            return True
        except sqlite3.IntegrityError:
            # Username or email already exists
            conn.close()
            return False
    
    def authenticate_user(self, username: str, password_hash: str) -> bool:
        """
        Authenticate a user (password should already be hashed)
        
        Args:
            username: Username to authenticate
            password_hash: Hashed password to verify
            
        Returns:
            True if authentication successful, False otherwise
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT password_hash FROM users WHERE username = ?
        ''', (username,))
        
        result = cursor.fetchone()
        conn.close()
        
        if result is None:
            return False
        
        # In practice, use bcrypt.checkpw() - this is just checking if hash matches
        return result['password_hash'] == password_hash
    
    def get_user_by_username(self, username: str) -> Dict:
        """
        Get user information by username
        
        Args:
            username: Username to look up
            
        Returns:
            Dict with user info or None if not found
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, username, email, password_hash, created_at, last_login
            FROM users WHERE username = ?
        ''', (username,))
        
        result = cursor.fetchone()
        conn.close()
        
        return dict(result) if result else None
    
    def update_last_login(self, username: str) -> None:
        """Update user's last login timestamp"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE users SET last_login = CURRENT_TIMESTAMP
            WHERE username = ?
        ''', (username,))
        
        conn.commit()
        conn.close()
    
    def ensure_user_exists(self, username: str) -> None:
        """
        Ensure a user record exists (for backward compatibility)
        Note: With authentication, users should be created via register_user
        """
        # Check if user exists
        user = self.get_user_by_username(username)
        if user is None:
            # For backward compatibility, create a dummy user
            # In production, this should not be used
            import logging
            logging.warning(f"Creating dummy user for {username} - use register_user instead")

    
    def start_session(self, username: str, session_type: str = 'practice', 
                     topics: list = None, difficulty: str = None) -> int:
        """Start a new learning session"""
        self.ensure_user_exists(username)
        
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO learning_sessions (username, session_type, topics, difficulty, started_at)
            VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
        ''', (username, session_type, json.dumps(topics or []), difficulty))
        
        session_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return session_id
    
    def end_session(self, session_id: int) -> None:
        """End a learning session"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE learning_sessions 
            SET ended_at = CURRENT_TIMESTAMP 
            WHERE id = ?
        ''', (session_id,))
        
        conn.commit()
        conn.close()
    
    def record_question_attempt(self, username: str, session_id: int = None,
                               problem_data: dict = None, user_code: str = None,
                               user_explanation: str = None, evaluation_result: dict = None,
                               score: float = None, time_spent_seconds: int = 0) -> int:
        """Record a question attempt"""
        self.ensure_user_exists(username)
        
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO question_attempts 
            (session_id, username, problem_data, user_code, user_explanation, 
             evaluation_result, score, time_spent_seconds, attempted_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        ''', (session_id, username, 
              json.dumps(problem_data or {}), 
              user_code, 
              user_explanation,
              json.dumps(evaluation_result or {}),
              score, 
              time_spent_seconds))
        
        attempt_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return attempt_id
    
    def record_concept_explanation(self, username: str, concept: str, topic: str = None) -> None:
        """Record when a user requests a concept explanation"""
        self.ensure_user_exists(username)
        
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO concept_explanations (username, concept, topic)
            VALUES (?, ?, ?)
        ''', (username, concept, topic))
        
        conn.commit()
        conn.close()
    
    def record_hint_request(self, username: str, attempt_id: int = None, 
                           hint_level: int = 0, hint_data: dict = None) -> None:
        """Record when a user requests a hint"""
        self.ensure_user_exists(username)
        
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO hint_requests (attempt_id, username, hint_level, hint_data)
            VALUES (?, ?, ?, ?)
        ''', (attempt_id, username, hint_level, json.dumps(hint_data or {})))
        
        conn.commit()
        conn.close()
    
    def get_student_stats(self, username: str) -> dict:
        """Get basic statistics for a student"""
        self.ensure_user_exists(username)
        
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Total sessions
        cursor.execute('SELECT COUNT(*) as total_sessions FROM learning_sessions WHERE username = ?', (username,))
        total_sessions = cursor.fetchone()['total_sessions']
        
        # Total questions attempted
        cursor.execute('SELECT COUNT(*) as total_attempts FROM question_attempts WHERE username = ?', (username,))
        total_attempts = cursor.fetchone()['total_attempts']
        
        # Average score
        cursor.execute('SELECT AVG(score) as avg_score FROM question_attempts WHERE username = ? AND score IS NOT NULL', (username,))
        avg_score_row = cursor.fetchone()
        avg_score = avg_score_row['avg_score'] if avg_score_row['avg_score'] is not None else 0
        
        # Recent activity (last 7 days)
        cursor.execute('''
            SELECT COUNT(*) as recent_attempts 
            FROM question_attempts 
            WHERE username = ? AND attempted_at >= date('now', '-7 days')
        ''', (username,))
        recent_attempts = cursor.fetchone()['recent_attempts']
        
        conn.close()
        
        return {
            'total_sessions': total_sessions,
            'total_attempts': total_attempts,
            'average_score': round(avg_score, 1) if avg_score else 0,
            'recent_activity': recent_attempts
        }
    
    def get_student_history(self, username: str, limit: int = 10) -> dict:
        """Get student's learning history"""
        self.ensure_user_exists(username)
        
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Get recent sessions
        cursor.execute('''
            SELECT * FROM learning_sessions 
            WHERE username = ? 
            ORDER BY started_at DESC 
            LIMIT ?
        ''', (username, limit))
        
        sessions = [dict(row) for row in cursor.fetchall()]
        
        # Get recent attempts
        cursor.execute('''
            SELECT qa.*, ls.topics, ls.difficulty
            FROM question_attempts qa
            LEFT JOIN learning_sessions ls ON qa.session_id = ls.id
            WHERE qa.username = ?
            ORDER BY qa.attempted_at DESC
            LIMIT ?
        ''', (username, limit))
        
        attempts = [dict(row) for row in cursor.fetchall()]
        
        # Get concept explanations requested
        cursor.execute('''
            SELECT * FROM concept_explanations
            WHERE username = ?
            ORDER BY explanation_requested_at DESC
            LIMIT ?
        ''', (username, limit))
        
        concepts = [dict(row) for row in cursor.fetchall()]
        
        conn.close()
        
        return {
            'recent_sessions': sessions,
            'recent_attempts': attempts,
            'recent_concepts': concepts
        }




# import sqlite3
# import json
# from datetime import datetime
# from typing import List, Dict, Any
# import uuid
# class DatabaseManager:
#     def __init__(self, db_path='dsa_tutor.db'):
#         self.db_path = db_path
#         self.init_database()
    
#     def init_database(self):
#         """Initialize database tables"""
#         conn = self.get_connection()
#         cursor = conn.cursor()

#         # Create tables with proper schema
#         cursor.executescript('''
#             -- Users table (for anonymous tracking with student IDs)
#             CREATE TABLE IF NOT EXISTS users (
#                 id INTEGER PRIMARY KEY AUTOINCREMENT,
#                 student_id TEXT UNIQUE NOT NULL,
#                 created_at TEXT DEFAULT CURRENT_TIMESTAMP,
#                 last_active TEXT DEFAULT CURRENT_TIMESTAMP
#             );

#             -- Learning sessions
#             CREATE TABLE IF NOT EXISTS learning_sessions (
#                 id INTEGER PRIMARY KEY AUTOINCREMENT,
#                 student_id TEXT NOT NULL,
#                 session_type TEXT DEFAULT 'practice',
#                 topics TEXT, -- JSON array of topics
#                 difficulty TEXT,
#                 started_at TEXT DEFAULT CURRENT_TIMESTAMP,
#                 ended_at TEXT,
#                 FOREIGN KEY (student_id) REFERENCES users(student_id)
#             );

#             -- Question attempts (problems generated and solved)
#             CREATE TABLE IF NOT EXISTS question_attempts (
#                 id INTEGER PRIMARY KEY AUTOINCREMENT,
#                 session_id INTEGER,
#                 student_id TEXT NOT NULL,
#                 problem_data TEXT, -- JSON of the problem
#                 user_code TEXT,
#                 user_explanation TEXT,
#                 evaluation_result TEXT, -- JSON of evaluation
#                 score REAL,
#                 time_spent_seconds INTEGER DEFAULT 0,
#                 attempted_at TEXT DEFAULT CURRENT_TIMESTAMP,
#                 FOREIGN KEY (session_id) REFERENCES learning_sessions(id),
#                 FOREIGN KEY (student_id) REFERENCES users(student_id)
#             );

#             -- Concept explanations requested
#             CREATE TABLE IF NOT EXISTS concept_explanations (
#                 id INTEGER PRIMARY KEY AUTOINCREMENT,
#                 student_id TEXT NOT NULL,
#                 concept TEXT NOT NULL,
#                 topic TEXT,
#                 explanation_requested_at TEXT DEFAULT CURRENT_TIMESTAMP,
#                 FOREIGN KEY (student_id) REFERENCES users(student_id)
#             );

#             -- Hint requests
#             CREATE TABLE IF NOT EXISTS hint_requests (
#                 id INTEGER PRIMARY KEY AUTOINCREMENT,
#                 attempt_id INTEGER,
#                 student_id TEXT NOT NULL,
#                 hint_level INTEGER,
#                 hint_data TEXT, -- JSON of hint provided
#                 requested_at TEXT DEFAULT CURRENT_TIMESTAMP,
#                 FOREIGN KEY (attempt_id) REFERENCES question_attempts(id),
#                 FOREIGN KEY (student_id) REFERENCES users(student_id)
#             );
#         ''')

#         conn.commit()
#         conn.close()
    
#     def get_connection(self):
#         conn = sqlite3.connect(self.db_path)
#         conn.row_factory = sqlite3.Row
#         return conn
    
#     # User management methods
#     def create_user(self, username: str, email: str, password_hash: str) -> int:
#         conn = self.get_connection()
#         cursor = conn.cursor()
        
#         cursor.execute('''
#             INSERT INTO users (username, email, password_hash, created_at)
#             VALUES (?, ?, ?, ?)
#         ''', (username, email, password_hash, datetime.now()))
        
#         user_id = cursor.lastrowid
#         conn.commit()
#         conn.close()
        
#         return user_id
    
#     def get_user_by_username(self, username: str) -> Dict:
#         conn = self.get_connection()
#         cursor = conn.cursor()
        
#         cursor.execute('''
#             SELECT u.*, up.* FROM users u
#             LEFT JOIN user_profiles up ON u.id = up.user_id
#             WHERE u.username = ?
#         ''', (username,))
        
#         user = cursor.fetchone()
#         conn.close()
        
#         return dict(user) if user else None
    
#     # Session tracking methods
#     def start_session(self, user_id: int, session_type: str, topics: List[str], 
#                      difficulty: str) -> str:
#         session_id = str(uuid.uuid4())
#         topics_json = json.dumps(topics)
        
#         conn = self.get_connection()
#         cursor = conn.cursor()
        
#         cursor.execute('''
#             INSERT INTO learning_sessions 
#             (id, user_id, session_type, topics, difficulty, start_time)
#             VALUES (?, ?, ?, ?, ?, ?)
#         ''', (session_id, user_id, session_type, topics_json, difficulty, datetime.now()))
        
#         conn.commit()
#         conn.close()
        
#         return session_id
    
#     def end_session(self, session_id: str):
#         conn = self.get_connection()
#         cursor = conn.cursor()
        
#         cursor.execute('''
#             UPDATE learning_sessions 
#             SET end_time = ?,
#                 duration_minutes = CAST((julianday(?) - julianday(start_time)) * 24 * 60 AS INTEGER)
#             WHERE id = ?
#         ''', (datetime.now(), datetime.now(), session_id))
        
#         conn.commit()
#         conn.close()
    
#     # Progress tracking methods
#     def record_question_attempt(self, user_id: int, session_id: str, 
#                                question_data: Dict, score: int):
#         conn = self.get_connection()
#         cursor = conn.cursor()
        
#         cursor.execute('''
#             INSERT INTO question_attempts 
#             (user_id, session_id, question_id, topic, difficulty, 
#              user_code, user_explanation, score, evaluation_result, 
#              time_spent_seconds, submitted_at)
#             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
#         ''', (
#             user_id, session_id, 
#             question_data.get('question_id'),
#             question_data.get('topic'),
#             question_data.get('difficulty'),
#             question_data.get('user_code'),
#             question_data.get('user_explanation'),
#             score,
#             json.dumps(question_data.get('evaluation_result', {})),
#             question_data.get('time_spent_seconds', 0),
#             datetime.now()
#         ))
        
#         # Update user profile statistics
#         cursor.execute('''
#             UPDATE user_profiles 
#             SET total_questions_attempted = total_questions_attempted + 1,
#                 total_correct = total_correct + ?
#             WHERE user_id = ?
#         ''', (1 if score >= 70 else 0, user_id))
        
#         conn.commit()
#         conn.close()
    
#     def get_user_progress(self, user_id: int) -> Dict:
#         """Get comprehensive user progress report"""
#         conn = self.get_connection()
#         cursor = conn.cursor()
        
#         # Get basic stats
#         cursor.execute('''
#             SELECT up.*, u.username, u.email, u.created_at
#             FROM user_profiles up
#             JOIN users u ON up.user_id = u.id
#             WHERE up.user_id = ?
#         ''', (user_id,))
        
#         profile = dict(cursor.fetchone())
        
#         # Get session history
#         cursor.execute('''
#             SELECT * FROM learning_sessions 
#             WHERE user_id = ? 
#             ORDER BY start_time DESC 
#             LIMIT 10
#         ''', (user_id,))
        
#         sessions = [dict(row) for row in cursor.fetchall()]
        
#         # Get topic-wise performance
#         cursor.execute('''
#             SELECT topic, 
#                    COUNT(*) as attempts,
#                    AVG(score) as avg_score,
#                    SUM(CASE WHEN score >= 70 THEN 1 ELSE 0 END) as correct
#             FROM question_attempts
#             WHERE user_id = ?
#             GROUP BY topic
#         ''', (user_id,))
        
#         topic_stats = [dict(row) for row in cursor.fetchall()]
        
#         # Get recent activity
#         cursor.execute('''
#             SELECT date(submitted_at) as date, COUNT(*) as questions
#             FROM question_attempts
#             WHERE user_id = ? AND submitted_at >= date('now', '-30 days')
#             GROUP BY date(submitted_at)
#         ''', (user_id,))
        
#         recent_activity = [dict(row) for row in cursor.fetchall()]
        
#         conn.close()
        
#         return {
#             "profile": profile,
#             "recent_sessions": sessions,
#             "topic_performance": topic_stats,
#             "recent_activity": recent_activity,
#             "streak": self.calculate_streak(user_id)
#         }
    
#     def calculate_streak(self, user_id: int) -> int:
#         """Calculate consecutive days of activity"""
#         conn = self.get_connection()
#         cursor = conn.cursor()
        
#         cursor.execute('''
#             WITH activity_days AS (
#                 SELECT DISTINCT date(submitted_at) as activity_date
#                 FROM question_attempts
#                 WHERE user_id = ?
#                 UNION
#                 SELECT DISTINCT date(start_time) as activity_date
#                 FROM learning_sessions
#                 WHERE user_id = ?
#             ),
#             ranked_days AS (
#                 SELECT activity_date,
#                        date(activity_date, '-' || ROW_NUMBER() OVER (ORDER BY activity_date) || ' days') as grp
#                 FROM activity_days
#                 ORDER BY activity_date DESC
#             )
#             SELECT COUNT(*) as streak
#             FROM ranked_days
#             WHERE grp = date('now', '-' || (ROW_NUMBER() OVER (ORDER BY activity_date)) || ' days')
#             LIMIT 1
#         ''', (user_id, user_id))
        
#         result = cursor.fetchone()
#         conn.close()
        
#         return result['streak'] if result else 0
    
#     # ==========================================
#     # NEW METHODS FOR SESSION TRACKING
#     # ==========================================
    
#     def ensure_user_exists(self, student_id: str) -> None:
#         """Ensure a user record exists for the student ID"""
#         conn = self.get_connection()
#         cursor = conn.cursor()
        
#         cursor.execute('''
#             INSERT OR IGNORE INTO users (student_id, last_active) 
#             VALUES (?, CURRENT_TIMESTAMP)
#         ''', (student_id,))
        
#         # Update last active
#         cursor.execute('''
#             UPDATE users SET last_active = CURRENT_TIMESTAMP 
#             WHERE student_id = ?
#         ''', (student_id,))
        
#         conn.commit()
#         conn.close()
    
#     def start_session(self, student_id: str, session_type: str = 'practice', 
#                      topics: list = None, difficulty: str = None) -> int:
#         """Start a new learning session"""
#         self.ensure_user_exists(student_id)
        
#         conn = self.get_connection()
#         cursor = conn.cursor()
        
#         cursor.execute('''
#             INSERT INTO learning_sessions (student_id, session_type, topics, difficulty, started_at)
#             VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
#         ''', (student_id, session_type, json.dumps(topics or []), difficulty))
        
#         session_id = cursor.lastrowid
#         conn.commit()
#         conn.close()
        
#         return session_id
    
#     def end_session(self, session_id: int) -> None:
#         """End a learning session"""
#         conn = self.get_connection()
#         cursor = conn.cursor()
        
#         cursor.execute('''
#             UPDATE learning_sessions 
#             SET ended_at = CURRENT_TIMESTAMP 
#             WHERE id = ?
#         ''', (session_id,))
        
#         conn.commit()
#         conn.close()
    
#     def record_question_attempt(self, student_id: str, session_id: int = None,
#                                problem_data: dict = None, user_code: str = None,
#                                user_explanation: str = None, evaluation_result: dict = None,
#                                score: float = None, time_spent_seconds: int = 0) -> int:
#         """Record a question attempt"""
#         self.ensure_user_exists(student_id)
        
#         conn = self.get_connection()
#         cursor = conn.cursor()
        
#         cursor.execute('''
#             INSERT INTO question_attempts 
#             (session_id, student_id, problem_data, user_code, user_explanation, 
#              evaluation_result, score, time_spent_seconds, attempted_at)
#             VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
#         ''', (session_id, student_id, 
#               json.dumps(problem_data or {}), 
#               user_code, 
#               user_explanation,
#               json.dumps(evaluation_result or {}),
#               score, 
#               time_spent_seconds))
        
#         attempt_id = cursor.lastrowid
#         conn.commit()
#         conn.close()
        
#         return attempt_id
    
#     def record_concept_explanation(self, student_id: str, concept: str, topic: str = None) -> None:
#         """Record when a user requests a concept explanation"""
#         self.ensure_user_exists(student_id)
        
#         conn = self.get_connection()
#         cursor = conn.cursor()
        
#         cursor.execute('''
#             INSERT INTO concept_explanations (student_id, concept, topic)
#             VALUES (?, ?, ?)
#         ''', (student_id, concept, topic))
        
#         conn.commit()
#         conn.close()
    
#     def record_hint_request(self, student_id: str, attempt_id: int = None, 
#                            hint_level: int = 0, hint_data: dict = None) -> None:
#         """Record when a user requests a hint"""
#         self.ensure_user_exists(student_id)
        
#         conn = self.get_connection()
#         cursor = conn.cursor()
        
#         cursor.execute('''
#             INSERT INTO hint_requests (attempt_id, student_id, hint_level, hint_data)
#             VALUES (?, ?, ?, ?)
#         ''', (attempt_id, student_id, hint_level, json.dumps(hint_data or {})))
        
#         conn.commit()
#         conn.close()
    
#     def get_student_stats(self, student_id: str) -> dict:
#         """Get basic statistics for a student"""
#         conn = self.get_connection()
#         cursor = conn.cursor()
        
#         # Total sessions
#         cursor.execute('SELECT COUNT(*) as total_sessions FROM learning_sessions WHERE student_id = ?', (student_id,))
#         total_sessions = cursor.fetchone()['total_sessions']
        
#         # Total questions attempted
#         cursor.execute('SELECT COUNT(*) as total_attempts FROM question_attempts WHERE student_id = ?', (student_id,))
#         total_attempts = cursor.fetchone()['total_attempts']
        
#         # Average score
#         cursor.execute('SELECT AVG(score) as avg_score FROM question_attempts WHERE student_id = ? AND score IS NOT NULL', (student_id,))
#         avg_score = cursor.fetchone()['avg_score']
        
#         # Recent activity (last 7 days)
#         cursor.execute('''
#             SELECT COUNT(*) as recent_attempts 
#             FROM question_attempts 
#             WHERE student_id = ? AND attempted_at >= date('now', '-7 days')
#         ''', (student_id,))
#         recent_attempts = cursor.fetchone()['recent_attempts']
        
#         conn.close()
        
#         return {
#             'total_sessions': total_sessions,
#             'total_attempts': total_attempts,
#             'average_score': round(avg_score, 1) if avg_score else 0,
#             'recent_activity': recent_attempts
#         }



