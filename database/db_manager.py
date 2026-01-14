import sqlite3
import json
from datetime import datetime
from typing import List, Dict, Any

class DatabaseManager:
    def __init__(self, db_path='dsa_tutor.db'):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database tables"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Create tables (using the schema from Phase 1)
        cursor.executescript('''
            CREATE TABLE IF NOT EXISTS users (...);
            CREATE TABLE IF NOT EXISTS user_profiles (...);
            -- ... all other tables
        ''')
        
        conn.commit()
        conn.close()
    
    def get_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn
    
    # User management methods
    def create_user(self, username: str, email: str, password_hash: str) -> int:
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO users (username, email, password_hash, created_at)
            VALUES (?, ?, ?, ?)
        ''', (username, email, password_hash, datetime.now()))
        
        user_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return user_id
    
    def get_user_by_username(self, username: str) -> Dict:
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT u.*, up.* FROM users u
            LEFT JOIN user_profiles up ON u.id = up.user_id
            WHERE u.username = ?
        ''', (username,))
        
        user = cursor.fetchone()
        conn.close()
        
        return dict(user) if user else None
    
    # Session tracking methods
    def start_session(self, user_id: int, session_type: str, topics: List[str], 
                     difficulty: str) -> str:
        session_id = str(uuid.uuid4())
        topics_json = json.dumps(topics)
        
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO learning_sessions 
            (id, user_id, session_type, topics, difficulty, start_time)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (session_id, user_id, session_type, topics_json, difficulty, datetime.now()))
        
        conn.commit()
        conn.close()
        
        return session_id
    
    def end_session(self, session_id: str):
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE learning_sessions 
            SET end_time = ?,
                duration_minutes = CAST((julianday(?) - julianday(start_time)) * 24 * 60 AS INTEGER)
            WHERE id = ?
        ''', (datetime.now(), datetime.now(), session_id))
        
        conn.commit()
        conn.close()
    
    # Progress tracking methods
    def record_question_attempt(self, user_id: int, session_id: str, 
                               question_data: Dict, score: int):
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO question_attempts 
            (user_id, session_id, question_id, topic, difficulty, 
             user_code, user_explanation, score, evaluation_result, 
             time_spent_seconds, submitted_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            user_id, session_id, 
            question_data.get('question_id'),
            question_data.get('topic'),
            question_data.get('difficulty'),
            question_data.get('user_code'),
            question_data.get('user_explanation'),
            score,
            json.dumps(question_data.get('evaluation_result', {})),
            question_data.get('time_spent_seconds', 0),
            datetime.now()
        ))
        
        # Update user profile statistics
        cursor.execute('''
            UPDATE user_profiles 
            SET total_questions_attempted = total_questions_attempted + 1,
                total_correct = total_correct + ?
            WHERE user_id = ?
        ''', (1 if score >= 70 else 0, user_id))
        
        conn.commit()
        conn.close()
    
    def get_user_progress(self, user_id: int) -> Dict:
        """Get comprehensive user progress report"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Get basic stats
        cursor.execute('''
            SELECT up.*, u.username, u.email, u.created_at
            FROM user_profiles up
            JOIN users u ON up.user_id = u.id
            WHERE up.user_id = ?
        ''', (user_id,))
        
        profile = dict(cursor.fetchone())
        
        # Get session history
        cursor.execute('''
            SELECT * FROM learning_sessions 
            WHERE user_id = ? 
            ORDER BY start_time DESC 
            LIMIT 10
        ''', (user_id,))
        
        sessions = [dict(row) for row in cursor.fetchall()]
        
        # Get topic-wise performance
        cursor.execute('''
            SELECT topic, 
                   COUNT(*) as attempts,
                   AVG(score) as avg_score,
                   SUM(CASE WHEN score >= 70 THEN 1 ELSE 0 END) as correct
            FROM question_attempts
            WHERE user_id = ?
            GROUP BY topic
        ''', (user_id,))
        
        topic_stats = [dict(row) for row in cursor.fetchall()]
        
        # Get recent activity
        cursor.execute('''
            SELECT date(submitted_at) as date, COUNT(*) as questions
            FROM question_attempts
            WHERE user_id = ? AND submitted_at >= date('now', '-30 days')
            GROUP BY date(submitted_at)
        ''', (user_id,))
        
        recent_activity = [dict(row) for row in cursor.fetchall()]
        
        conn.close()
        
        return {
            "profile": profile,
            "recent_sessions": sessions,
            "topic_performance": topic_stats,
            "recent_activity": recent_activity,
            "streak": self.calculate_streak(user_id)
        }
    
    def calculate_streak(self, user_id: int) -> int:
        """Calculate consecutive days of activity"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            WITH activity_days AS (
                SELECT DISTINCT date(submitted_at) as activity_date
                FROM question_attempts
                WHERE user_id = ?
                UNION
                SELECT DISTINCT date(start_time) as activity_date
                FROM learning_sessions
                WHERE user_id = ?
            ),
            ranked_days AS (
                SELECT activity_date,
                       date(activity_date, '-' || ROW_NUMBER() OVER (ORDER BY activity_date) || ' days') as grp
                FROM activity_days
                ORDER BY activity_date DESC
            )
            SELECT COUNT(*) as streak
            FROM ranked_days
            WHERE grp = date('now', '-' || (ROW_NUMBER() OVER (ORDER BY activity_date)) || ' days')
            LIMIT 1
        ''', (user_id, user_id))
        
        result = cursor.fetchone()
        conn.close()
        
        return result['streak'] if result else 0