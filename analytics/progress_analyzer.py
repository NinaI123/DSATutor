import json
from datetime import datetime, timedelta
from typing import Dict, List, Any

class ProgressAnalyzer:
    def __init__(self, db_manager):
        self.db = db_manager
    
    def generate_weekly_report(self, user_id: int) -> Dict:
        """Generate weekly progress report"""
        conn = self.db.get_connection()
        cursor = conn.cursor()
        
        # Get this week's data
        week_start = datetime.now() - timedelta(days=7)
        
        cursor.execute('''
            SELECT 
                COUNT(DISTINCT date(start_time)) as active_days,
                COUNT(*) as total_sessions,
                AVG(duration_minutes) as avg_session_length
            FROM learning_sessions
            WHERE user_id = ? AND start_time >= ?
        ''', (user_id, week_start))
        
        session_stats = dict(cursor.fetchone())
        
        cursor.execute('''
            SELECT 
                COUNT(*) as questions_attempted,
                AVG(score) as avg_score,
                SUM(CASE WHEN score >= 70 THEN 1 ELSE 0 END) as correct_answers
            FROM question_attempts
            WHERE user_id = ? AND submitted_at >= ?
        ''', (user_id, week_start))
        
        question_stats = dict(cursor.fetchone())
        
        # Get improvement trends
        cursor.execute('''
            WITH daily_scores AS (
                SELECT date(submitted_at) as day, AVG(score) as daily_avg
                FROM question_attempts
                WHERE user_id = ? AND submitted_at >= date('now', '-14 days')
                GROUP BY date(submitted_at)
            )
            SELECT 
                AVG(CASE WHEN day >= date('now', '-7 days') THEN daily_avg END) as current_week_avg,
                AVG(CASE WHEN day < date('now', '-7 days') THEN daily_avg END) as previous_week_avg
            FROM daily_scores
        ''', (user_id,))
        
        trend = dict(cursor.fetchone())
        improvement = 0
        if trend['previous_week_avg']:
            improvement = ((trend['current_week_avg'] or 0) - trend['previous_week_avg']) / trend['previous_week_avg'] * 100
        
        conn.close()
        
        return {
            "period": "Weekly Report",
            "date_range": f"{week_start.date()} to {datetime.now().date()}",
            "session_stats": session_stats,
            "question_stats": question_stats,
            "improvement_percentage": f"{improvement:.1f}%",
            "recommendations": self.generate_recommendations(session_stats, question_stats)
        }
    
    def generate_recommendations(self, session_stats: Dict, question_stats: Dict) -> List[str]:
        """Generate personalized recommendations"""
        recommendations = []
        
        if session_stats['active_days'] < 3:
            recommendations.append("Try to practice at least 3 days a week for better retention")
        
        if question_stats.get('avg_score', 0) < 60:
            recommendations.append("Focus on reviewing incorrect questions and understanding the solutions")
        
        if session_stats.get('avg_session_length', 0) > 60:
            recommendations.append("Consider shorter, more frequent sessions for better focus")
        
        return recommendations
    
    def export_progress_data(self, user_id: int, format: str = "json") -> Any:
        """Export user progress data"""
        progress_data = self.db.get_user_progress(user_id)
        
        if format == "json":
            return json.dumps(progress_data, indent=2, default=str)
        elif format == "csv":
            # Convert to CSV format
            import csv
            import io
            
            output = io.StringIO()
            writer = csv.writer(output)
            
            # Write headers and data
            writer.writerow(["Metric", "Value"])
            for key, value in progress_data['profile'].items():
                if not isinstance(value, (dict, list)):
                    writer.writerow([key, value])
            
            return output.getvalue()
        
        return None