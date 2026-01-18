#!/usr/bin/env python3
"""
Main entry point for DSA Tutor System
Run this file to start the application
"""

import os
import sys
import logging
from pathlib import Path
import random

# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import configuration
from config import (
    GROQ_API_KEY, APP_NAME, APP_VERSION, DEBUG_MODE,
    SERVER_HOST, SERVER_PORT, SHARE_PUBLICLY, LOG_LEVEL,
    LOG_FILE, ENABLE_CONSOLE_LOG, print_config_summary
)

# Import core components
from orchestrator import DSATutorSystem
from database.db_manager import DatabaseManager
from utils.auth import hash_password, verify_password, validate_password, validate_email, validate_username
import gradio as gr


# ==================== SETUP LOGGING ====================
def setup_logging():
    """Configure logging for the application"""
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    handlers = []

    # File handler
    if LOG_FILE:
        file_handler = logging.FileHandler(log_dir / LOG_FILE)
        file_handler.setFormatter(logging.Formatter(log_format))
        handlers.append(file_handler)

    # Console handler
    if ENABLE_CONSOLE_LOG:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter(log_format))
        handlers.append(console_handler)

    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL),
        format=log_format,
        handlers=handlers
    )

    # Silence some noisy loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)


# ==================== CREATE GRADIO INTERFACE ====================
def create_tutor_interface(tutor: DSATutorSystem, db: DatabaseManager):
    """
    Create the Gradio web interface for the DSA Tutor with database tracking
    """
    from config import UI_THEME, DEFAULT_TOPICS, DEFAULT_DIFFICULTY

    # Custom CSS for better UI
    custom_css = """
    .gradio-container {
        max-width: 1200px !important;
        margin: 0 auto !important;
    }
    .agent-panel {
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 10px;
        margin: 10px 0;
    }
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 5px;
        padding: 10px;
        margin: 10px 0;
    }
    """

    with gr.Blocks(
        title=f"{APP_NAME} v{APP_VERSION}",
        theme=UI_THEME,
        css=custom_css
    ) as demo:
        
        # Session state (stored in Gradio state)
        current_session_id = gr.State(None)
        logged_in_user = gr.State(None)  # Stores username when logged in
        is_authenticated = gr.State(False)  # Authentication status
        current_problem = gr.State(None)
        current_attempt_id = gr.State(None)

        # Header
        gr.Markdown(f"""
        # {APP_NAME} v{APP_VERSION}
        ### Intelligent Data Structures & Algorithms Tutor
        *Four AI agents work together to teach you DSA concepts*
        """)

        # Status indicator
        status = gr.Textbox(
            label="System Status",
            value="System initialized and ready",
            interactive=False
        )

        with gr.Tabs() as tabs:
            # ========== TAB 0: AUTHENTICATION ==========
            with gr.TabItem("🔐 Login / Sign Up", id=0):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### Welcome to DSA Tutor!")
                        gr.Markdown("Please login or create an account to continue")
                        
                        # Toggle between login and signup
                        auth_mode = gr.Radio(
                            choices=["Login", "Sign Up"],
                            value="Login",
                            label="Choose Action"
                        )
                        
                        # Login form
                        with gr.Group(visible=True) as login_form:
                            gr.Markdown("#### Login")
                            login_username = gr.Textbox(
                                label="Username",
                                placeholder="Enter your username"
                            )
                            login_password = gr.Textbox(
                                label="Password",
                                type="password",
                                placeholder="Enter your password"
                            )
                            login_btn = gr.Button("Login", variant="primary")
                        
                        # Signup form
                        with gr.Group(visible=False) as signup_form:
                            gr.Markdown("#### Create Account")
                            signup_username = gr.Textbox(
                                label="Username",
                                placeholder="Choose a username (3-20 characters)"
                            )
                            signup_email = gr.Textbox(
                                label="Email",
                                placeholder="your.email@example.com"
                            )
                            signup_password = gr.Textbox(
                                label="Password",
                                type="password",
                                placeholder="Min 8 chars, 1 uppercase, 1 lowercase, 1 digit"
                            )
                            signup_confirm = gr.Textbox(
                                label="Confirm Password",
                                type="password",
                                placeholder="Re-enter your password"
                            )
                            signup_btn = gr.Button("Sign Up", variant="primary")
                    
                    with gr.Column(scale=1):
                        gr.Markdown("### System Information")
                        auth_status = gr.Markdown("**Status:** Not logged in")
                        auth_message = gr.Markdown("")
                        
                        gr.Markdown("""
                        ### Test Account
                        For testing, you can use:
                        - **Username:** testuser
                        - **Password:** TestPass123
                        
                        ### Password Requirements
                        - At least 8 characters
                        - One uppercase letter
                        - One lowercase letter
                        - One digit
                        """)

        with gr.Tabs(visible=False) as main_tabs:
            # ========== TAB 1: LEARNING SESSION ==========
            with gr.TabItem("🎓 Learning Session", id=1):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### Learning Settings")

                        topics = gr.CheckboxGroup(
                            choices=["Arrays", "Linked Lists", "Trees",
                                   "Graphs", "Sorting", "Searching",
                                   "Dynamic Programming", "Recursion", "Backtracking"],
                            value=DEFAULT_TOPICS,
                            label="Select Topics to Learn"
                        )

                        difficulty = gr.Dropdown(
                            choices=["Easy", "Medium", "Hard"],
                            value=DEFAULT_DIFFICULTY,
                            label="Difficulty Level"
                        )

                        start_btn = gr.Button(
                            "🚀 Start Learning Session",
                            variant="primary",
                            size="lg"
                        )

                    with gr.Column(scale=2):
                        gr.Markdown("### Session Information")
                        session_output = gr.JSON(label="Session Details")
                        welcome_msg = gr.Markdown(
                            label="Welcome Message",
                            value="*Start a session to begin learning*"
                        )

                with gr.Row():
                    gr.Markdown("### Learn a Concept")
                    concept_input = gr.Textbox(
                        label="Concept to Learn",
                        placeholder="e.g., Binary Search, Dynamic Programming, BFS vs DFS...",
                        scale=3
                    )
                    topic_select = gr.Dropdown(
                        choices=["Arrays", "Linked Lists", "Trees", "Graphs",
                               "Sorting", "Searching", "Dynamic Programming"],
                        value="Arrays",
                        label="Topic",
                        scale=1
                    )
                    explain_btn = gr.Button("📚 Explain Concept", variant="secondary")

                with gr.Row():
                    concept_output = gr.JSON(label="Concept Explanation")
                    concept_display = gr.Markdown(label="Detailed Explanation")

            # ========== TAB 2: PRACTICE PROBLEMS ==========
            with gr.TabItem("💪 Practice Problems", id=2):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### Problem Settings")
                        practice_topic = gr.Dropdown(
                            choices=["Arrays", "Linked Lists", "Trees", "Graphs",
                                   "Sorting", "Searching", "Dynamic Programming"],
                            value="Arrays",
                            label="Topic"
                        )

                        practice_difficulty = gr.Dropdown(
                            choices=["Easy", "Medium", "Hard"],
                            value="Medium",
                            label="Difficulty"
                        )

                        weaknesses = gr.Textbox(
                            label="Your Weak Areas (optional)",
                            placeholder="e.g., edge cases, time complexity, recursion...",
                            lines=2
                        )

                        generate_btn = gr.Button(
                            "🎲 Generate Practice Problem",
                            variant="primary"
                        )

                    with gr.Column(scale=2):
                        gr.Markdown("### Problem Details")
                        problem_display = gr.JSON(label="Generated Problem")
                        problem_desc = gr.Markdown(
                            label="Problem Description",
                            value="*Generate a problem to see it here*"
                        )

                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### Your Solution")
                        hint_level = gr.Slider(
                            0, 3, value=0, step=1,
                            label="Hint Level",
                            info="0=General hint, 3=Specific help"
                        )

                        with gr.Row():
                            get_hint_btn = gr.Button("💡 Get Hint", variant="secondary")
                            reset_hints_btn = gr.Button("🔄 Reset Hints")

                        student_code = gr.Code(
                            label="Your Python Code",
                            value="def solution():\n    # Write your solution here\n    pass",
                            language="python",
                            lines=10
                        )

                        student_explanation = gr.Textbox(
                            label="Explain Your Approach",
                            placeholder="Describe your thought process and algorithm...",
                            lines=3
                        )

                        evaluate_btn = gr.Button("✅ Evaluate Solution", variant="primary")

                    with gr.Column(scale=1):
                        gr.Markdown("### Feedback")
                        hint_output = gr.JSON(label="Hint")
                        evaluation_output = gr.JSON(label="Evaluation Results")
                        evaluation_display = gr.Markdown(label="Check detailed feedback here")

            # ========== TAB 3: PROGRESS TRACKING ==========
            with gr.TabItem("📈 Progress & Analytics", id=3):
                with gr.Row():
                    progress_id = gr.Textbox(
                        label="Student ID",
                        value="student_001",
                        info="Enter student ID to view progress"
                    )
                    get_progress_btn = gr.Button("📊 Get Progress Report", variant="primary")

                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### 📊 Your Statistics")
                        stats_display = gr.JSON(label="Learning Statistics")
                    
                    with gr.Column():
                        gr.Markdown("### 🎯 Recent Activity")
                        recent_activity = gr.Markdown(label="Activity Summary", value="*Click 'Get Progress Report' to load your stats*")

                with gr.Row():
                    progress_report = gr.JSON(label="Detailed Progress")
                    learning_path = gr.JSON(label="Recommended Learning Path")

            # ========== TAB 4: AGENT DASHBOARD ==========
            with gr.TabItem("🤖 Agent Dashboard", id=4):
                gr.Markdown("## Multi-Agent System Status")

                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### 👨‍🏫 Teacher Agent")
                        teacher_status = gr.Textbox(
                            label="Status",
                            value="🟢 Active - Ready to teach",
                            interactive=False
                        )
                        teacher_stats = gr.Textbox(
                            label="Statistics",
                            value="Concepts explained: 0\nSessions conducted: 0",
                            interactive=False
                        )

                    with gr.Column():
                        gr.Markdown("### ❓ Question Generator")
                        qgen_status = gr.Textbox(
                            label="Status",
                            value="🟢 Active - Ready to generate problems",
                            interactive=False
                        )
                        qgen_stats = gr.Textbox(
                            label="Statistics",
                            value="Problems generated: 0\nDifficulty distribution: N/A",
                            interactive=False
                        )

                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### 💡 Hint Agent")
                        hint_status = gr.Textbox(
                            label="Status",
                            value="🟢 Active - Ready to provide hints",
                            interactive=False
                        )
                        hint_stats = gr.Textbox(
                            label="Statistics",
                            value="Hints given: 0\nAverage hint level: 0.0",
                            interactive=False
                        )

                    with gr.Column():
                        gr.Markdown("### ✅ Evaluator Agent")
                        eval_status = gr.Textbox(
                            label="Status",
                            value="🟢 Active - Ready to evaluate solutions",
                            interactive=False
                        )
                        eval_stats = gr.Textbox(
                            label="Statistics",
                            value="Solutions evaluated: 0\nAverage score: 0.0",
                            interactive=False
                        )

                with gr.Row():
                    refresh_btn = gr.Button("🔄 Refresh Agent Status")
                    
        # ==================== AUTHENTICATION EVENT HANDLERS ====================
        
        def toggle_auth_forms(mode):
            """Toggle between login and signup forms"""
            if mode == "Login":
                return gr.update(visible=True), gr.update(visible=False)
            else:
                return gr.update(visible=False), gr.update(visible=True)
        
        def handle_login(username, password):
            """Handle user login"""
            if not username or not password:
                return (
                    "**Status:** Login failed",
                    "❌ Please enter both username and password",
                    None,  # logged_in_user
                    False,  # is_authenticated
                    gr.update(visible=False),  # main_tabs
                    gr.update(visible=True)  # auth tab
                )
            
            # Hash the password to compare
            user = db.get_user_by_username(username)
            if user and verify_password(password, user['password_hash']):
                # Successful login
                db.update_last_login(username)
                return (
                    f"**Status:** Logged in as **{username}**",
                    f"✅ Welcome back, {username}!",
                    username,  # logged_in_user
                    True,  # is_authenticated
                    gr.update(visible=True),  # main_tabs
                    gr.update(visible=False)  # auth tab (hide it)
                )
            else:
                return (
                    "**Status:** Login failed",
                    "❌ Invalid username or password",
                    None,
                    False,
                    gr.update(visible=False),
                    gr.update(visible=True)
                )
        
        def handle_signup(username, email, password, confirm_password):
            """Handle user registration"""
            # Validate inputs
            if not username or not email or not password or not confirm_password:
                return (
                    "**Status:** Signup failed",
                    "❌ All fields are required",
                    None,
                    False,
                    gr.update(visible=False),
                    gr.update(visible=True)
                )
            
            # Validate username
            valid_username, username_error = validate_username(username)
            if not valid_username:
                return (
                    "**Status:** Signup failed",
                    f"❌ {username_error}",
                    None,
                    False,
                    gr.update(visible=False),
                    gr.update(visible=True)
                )
            
            # Validate email
            valid_email, email_error = validate_email(email)
            if not valid_email:
                return (
                    "**Status:** Signup failed",
                    f"❌ {email_error}",
                    None,
                    False,
                    gr.update(visible=False),
                    gr.update(visible=True)
                )
            
            # Validate password
            valid_password, password_error = validate_password(password)
            if not valid_password:
                return (
                    "**Status:** Signup failed",
                    f"❌ {password_error}",
                    None,
                    False,
                    gr.update(visible=False),
                    gr.update(visible=True)
                )
            
            # Check password confirmation
            if password != confirm_password:
                return (
                    "**Status:** Signup failed",
                    "❌ Passwords do not match",
                    None,
                    False,
                    gr.update(visible=False),
                    gr.update(visible=True)
                )
            
            # Hash password and register user
            hashed_password = hash_password(password)
            success = db.register_user(username, email, hashed_password)
            
            if success:
                # Auto-login after successful registration
                db.update_last_login(username)
                return (
                    f"**Status:** Logged in as **{username}**",
                    f"✅ Account created successfully! Welcome, {username}!",
                    username,
                    True,
                    gr.update(visible=True),
                    gr.update(visible=False)
                )
            else:
                return (
                    "**Status:** Signup failed",
                    "❌ Username or email already exists",
                    None,
                    False,
                    gr.update(visible=False),
                    gr.update(visible=True)
                )
        
        # ==================== EVENT HANDLERS ====================
        def start_session(student_id, topics, difficulty):
            try:
                if not student_id.strip():
                    student_id = "default_student_001"
                    
                # Update current student ID
                student_id = student_id.strip()
                
                # Start database session
                session_id = db.start_session(student_id, 'practice', topics, difficulty)
                
                # Start tutor session
                session = tutor.start_learning_session(student_id, topics, difficulty)
                
                # Format welcome message with learning plan
                welcome_msg = session.get("welcome_message", "")
                learning_plan = session.get("learning_plan", [])
                
                if learning_plan:
                    welcome_msg += "\n\n### 🗓️ Your Session Plan\n"
                    for item in learning_plan:
                        icon_map = {
                            "Warmup": "🌅", 
                            "Concept Exploration": "📘", 
                            "Concept": "📘", 
                            "Guided Practice": "🤝", 
                            "Independent Practice": "💪", 
                            "Practice": "💪",
                            "Reflection": "💭"
                        }
                        phase = item.get('phase', 'Activity')
                        icon = icon_map.get(phase, "📌")
                        
                        welcome_msg += (
                            f"**{icon} {phase}** ({item.get('estimated_time', '5m')})\n"
                            f"> *{item.get('description', '')}*\n\n"
                        )

                return (
                    {"student_id": student_id, "session_id": session_id, "topics": topics, "difficulty": difficulty},
                    welcome_msg,
                    f"✅ Session started for {student_id}",
                    session_id,
                    student_id  # Return the student ID to update the state
                )
            except Exception as e:
                logging.error(f"Error starting session: {e}")
                return {}, f"Error: {str(e)}", f"❌ Error: {str(e)}", None, student_id

        def explain_concept(concept, topic, username):
            if not concept.strip():
                return {}, "Please enter a concept to learn", "⚠️ Please enter a concept to learn"
            try:
                if not username:
                    return {}, "Error: Not logged in", "❌ Please login first"
                    
                explanation = tutor.get_concept_explanation(concept, topic)
                
                # Track concept explanation request
                db.record_concept_explanation(username, concept, topic)
                
                explanation_text = f"""
## {explanation.get('title', concept)}

**Concept:** {concept}  
**Topic:** {topic}  
**Difficulty Level:** {explanation.get('difficulty', 'Intermediate')}

### Explanation
{explanation.get('explanation', '')}

### Key Points
{explanation.get('key_points', '')}

### Examples
{explanation.get('examples', '')}
"""
                
                return explanation, explanation_text, f"✅ Explained '{concept}' in {topic}"
            except Exception as e:
                logging.error(f"Error explaining concept: {e}")
                return {}, f"Error: {str(e)}", f"❌ Error: {str(e)}"

        def generate_problem(topic, difficulty, weaknesses, student_id):
            try:
                if not student_id:
                    student_id = "default_student_001"
                    
                weakness_list = [w.strip() for w in weaknesses.split(",")] if weaknesses else []
                problem = tutor.generate_practice_question(topic, difficulty, weakness_list)
                
                problem_text = f"""
## {problem.get('title', 'Practice Problem')}

**Topic:** {topic}  
**Difficulty:** {difficulty}  
**Time Estimate:** {problem.get('time_estimate', '15-20 minutes')}

### Problem Statement
{problem.get('description', '')}

### Input Format
{problem.get('input_format', '')}

### Output Format
{problem.get('output_format', '')}

### Constraints
{problem.get('constraints', '')}
"""
                
                return problem, problem_text, f"🎲 Generated {difficulty} problem on {topic}"
            except Exception as e:
                logging.error(f"Error generating problem: {e}")
                return {}, f"Error: {str(e)}", f"❌ Error: {str(e)}"

        def get_hint(problem, hint_level, student_id, attempt_id):
            if not problem:
                return {"error": "No problem provided"}, "⚠️ No problem provided"
            try:
                if not student_id:
                    student_id = "default_student_001"
                    
                hint = tutor.get_hint(problem, hint_level)
                
                # Track hint request
                if attempt_id:
                    db.record_hint_request(student_id, attempt_id, hint_level, hint)
                
                hint_text = f"""
## Hint Level {hint['hint_level']}

**Focus Area:** {hint.get('focus_area', 'General')}

### Hint
{hint.get('hint', '')}

### Thinking Direction
{hint.get('thinking_direction', '')}

**Remember:** Try to solve it yourself before asking for the next hint!
"""
                
                return hint, hint_text, f"💡 Hint level {hint['hint_level']} provided"
            except Exception as e:
                logging.error(f"Error getting hint: {e}")
                return {"error": str(e)}, f"Error: {str(e)}", f"❌ Error: {str(e)}"

        def evaluate_solution(problem, code, explanation, student_id, session_id):
            if not problem or not code:
                return {"error": "Problem and code required"}, "⚠️ Problem and code required", None
            try:
                if not student_id:
                    student_id = "default_student_001"
                    
                evaluation = tutor.evaluate_solution(problem, code, explanation)
                score = evaluation.get('score', 0)
              
                # Track question attempt
                attempt_id = db.record_question_attempt(
                    username=student_id,
                    session_id=session_id,
                    problem_data=problem,
                    user_code=code,
                    user_explanation=explanation,
                    evaluation_result=evaluation,
                    score=score
                )
                
                evaluation_text = f"""
## Evaluation Results

**Score:** {score}/100

### Feedback
{evaluation.get('feedback', '')}

### Correctness
{evaluation.get('correctness', '')}

### Suggestions for Improvement
{evaluation.get('improvement_suggestions', '')}

### Time Complexity
{evaluation.get('time_complexity', '')}

### Space Complexity
{evaluation.get('space_complexity', '')}
"""
                
                return evaluation, evaluation_text, f"✅ Solution evaluated: {score}/100", attempt_id
            except Exception as e:
                logging.error(f"Error evaluating solution: {e}")
                return {"error": str(e)}, f"Error: {str(e)}", f"❌ Error: {str(e)}", None

        def get_progress_report(student_id):
            try:
                if not student_id:
                    student_id = "default_student_001"
                    
                # Get statistics from database
                stats = db.get_student_stats(student_id)
                
                # Get history
                history = db.get_student_history(student_id, limit=5)
                
                # Format progress report
                progress = {
                    "student_id": student_id,
                    "total_sessions": stats['total_sessions'],
                    "total_attempts": stats['total_attempts'],
                    "average_score": stats['average_score'],
                    "recent_activity": f"{stats['recent_activity']} attempts in last 7 days",
                    "recent_sessions": len(history['recent_sessions']),
                    "recent_concepts": len(history['recent_concepts'])
                }
                
                # Format activity summary
                activity_text = f"""
## 📊 Progress Report for {student_id}

### 📈 Statistics
- **Total Sessions:** {stats['total_sessions']}
- **Problems Solved:** {stats['total_attempts']}
- **Average Score:** {stats['average_score']:.1f}%
- **Recent Activity:** {stats['recent_activity']} attempts this week

### 📚 Recent Learning
- **Recent Sessions:** {len(history['recent_sessions'])} in history
- **Concepts Learned:** {len(history['recent_concepts'])} recently

### 🎯 Recommendations
1. **Consistency is Key:** Try to practice daily
2. **Focus on Weak Areas:** Review problems with lower scores
3. **Build Foundation:** Master basic concepts before advanced topics

**Keep up the great work! 🚀**
"""
                
                return progress, activity_text, f"📊 Progress report generated for {student_id}"
            except Exception as e:
                logging.error(f"Error getting progress: {e}")
                return {"error": str(e)}, f"Error loading statistics: {str(e)}", f"❌ Error: {str(e)}"

        # ==================== AUTHENTICATION BUTTON HANDLERS ====================
        
        # Toggle between login and signup forms
        auth_mode.change(
            toggle_auth_forms,
            inputs=[auth_mode],
            outputs=[login_form, signup_form]
        )
        
        # Login button
        login_btn.click(
            handle_login,
            inputs=[login_username, login_password],
            outputs=[auth_status, auth_message, logged_in_user, is_authenticated, main_tabs, tabs]
        )
        
        # Signup button
        signup_btn.click(
            handle_signup,
            inputs=[signup_username, signup_email, signup_password, signup_confirm],
            outputs=[auth_status, auth_message, logged_in_user, is_authenticated, main_tabs, tabs]
        )
        
        # ==================== MAIN APP BUTTON HANDLERS ====================
        
        # Connect event handlers with updated outputs
        start_btn.click(
            start_session,
            inputs=[logged_in_user, topics, difficulty],
            outputs=[session_output, welcome_msg, status, current_session_id, logged_in_user]
        )
        
        explain_btn.click(
            explain_concept,
            inputs=[concept_input, topic_select, logged_in_user],
            outputs=[concept_output, concept_display, status]
        )
        
        generate_btn.click(
            lambda topic, difficulty, weaknesses, username: generate_problem(topic, difficulty, weaknesses, username),
            inputs=[practice_topic, practice_difficulty, weaknesses, logged_in_user],
            outputs=[problem_display, problem_desc, status]
        )

        get_hint_btn.click(
            get_hint,
            inputs=[problem_display, hint_level, logged_in_user, current_attempt_id],
            outputs=[hint_output, status]
        )

        evaluate_btn.click(
            evaluate_solution,
            inputs=[problem_display, student_code, student_explanation, logged_in_user, current_session_id],
            outputs=[evaluation_output, evaluation_display, status, current_attempt_id]
        )

        get_progress_btn.click(
            get_progress_report,
            inputs=[progress_id],
            outputs=[progress_report, recent_activity, status]
        )
        # ==================== EVENT HANDLERS ====================
#         def start_session(student_id, topics, difficulty):
#             try:
#                 # Start database session
#                 session_id = db.start_session(student_id, 'practice', topics, difficulty)
                
#                 # Start tutor session
#                 session = tutor.start_learning_session(student_id, topics, difficulty)
                
#                 return session, session.get("welcome_message", ""), "✅ Session started successfully", session_id
#             except Exception as e:
#                 logging.error(f"Error starting session: {e}")
#                 return {}, f"Error starting session: {str(e)}", f"❌ Error: {str(e)}", None

#         def explain_concept(concept, topic, student_id):
#             if not concept.strip():
#                 return {}, "Please enter a concept to learn", "⚠️ Please enter a concept to learn"
#             try:
#                 explanation = tutor.get_concept_explanation(concept, topic)
                
#                 # Track concept explanation request
#                 db.record_concept_explanation(student_id, concept, topic)
                
#                 return explanation, explanation.get("explanation", ""), f"✅ Explained '{concept}' in {topic}"
#             except Exception as e:
#                 logging.error(f"Error explaining concept: {e}")
#                 return {}, f"Error explaining concept: {str(e)}", f"❌ Error: {str(e)}"

#         def generate_problem(topic, difficulty, weaknesses):
#             try:
#                 weakness_list = [w.strip() for w in weaknesses.split(",")] if weaknesses else []
#                 problem = tutor.generate_practice_question(topic, difficulty, weakness_list)
#                 return problem, f"### {problem.get('title', 'Problem')}\n\n{problem.get('description', '')}", f"🎲 Generated {difficulty} problem on {topic}"
#             except Exception as e:
#                 logging.error(f"Error generating problem: {e}")
#                 return {}, f"Error generating problem: {str(e)}", f"❌ Error: {str(e)}"

#         def get_hint(problem, hint_level, student_id, attempt_id):
#             if not problem:
#                 return {"error": "No problem provided"}, "⚠️ No problem provided"
#             try:
#                 hint = tutor.get_hint(problem, hint_level)
                
#                 # Track hint request
#                 if attempt_id:
#                     db.record_hint_request(student_id, attempt_id, hint_level, hint)
                
#                 return hint, f"💡 Hint level {hint['hint_level']} provided"
#             except Exception as e:
#                 logging.error(f"Error getting hint: {e}")
#                 return {"error": str(e)}, f" Error: {str(e)}"

#         def evaluate_solution(problem, code, explanation, student_id, session_id):
#             if not problem or not code:
#                 return {"error": "Problem and code required"}, " Problem and code required"
#             try:
#                 evaluation = tutor.evaluate_solution(problem, code, explanation)
#                 score = evaluation.get('score', 0)
                
#                 # Track question attempt
#                 attempt_id = db.record_question_attempt(
#                     student_id=student_id,
#                     session_id=session_id,
#                     problem_data=problem,
#                     user_code=code,
#                     user_explanation=explanation,
#                     evaluation_result=evaluation,
#                     score=score
#                 )
                
#                 return evaluation, f" Solution evaluated: {score}/100", attempt_id
#             except Exception as e:
#                 logging.error(f"Error evaluating solution: {e}")
#                 return {"error": str(e)}, f"Error: {str(e)}", None

#         def get_progress_report(student_id):
#             try:
#                 # Get statistics from database
#                 stats = db.get_student_stats(student_id)
                
#                 # Format progress report
#                 progress = {
#                     "student_id": student_id,
#                     "total_sessions": stats['total_sessions'],
#                     "total_attempts": stats['total_attempts'],
#                     "average_score": stats['average_score'],
#                     "recent_activity": f"{stats['recent_attempts']} attempts in last 7 days",
#                     "profile": {
#                         "student_id": student_id,
#                         "total_sessions": stats['total_sessions'],
#                         "total_questions": stats['total_attempts'],
#                         "average_score": stats['average_score']
#                     },
#                     "topic_performance": [],  # Could be enhanced later
#                     "recent_activity": []  # Could be enhanced later
#                 }
                
#                 # Format activity summary
#                 activity_text = f"""
# **Your Learning Activity:**
# - **Total Sessions:** {stats['total_sessions']}
# - **Problems Solved:** {stats['total_attempts']}
# - **Average Score:** {stats['average_score']:.1f}%
# - **Recent Activity:** {stats['recent_attempts']} attempts this week

# **Keep practicing to improve your skills! 🚀**
# """
                
#                 return progress, activity_text, f"📊 Progress report generated for {student_id}"
#             except Exception as e:
#                 logging.error(f"Error getting progress: {e}")
#                 return {"error": str(e)}, "*Error loading statistics*", f"❌ Error: {str(e)}"

#         # Connect event handlers
#         start_btn.click(
#             start_session,
#             inputs=[student_id, topics, difficulty],
#             outputs=[session_output, welcome_msg, status, current_session_id]
#         )
        
#         explain_btn.click(
#             explain_concept,
#             inputs=[concept_input, topic_select, current_student_id],
#             outputs=[concept_output, concept_display, status]
#         )
        
#         generate_btn.click(
#             generate_problem,
#             inputs=[practice_topic, practice_difficulty, weaknesses],
#             outputs=[problem_display, problem_desc, status]
#         )

#         get_hint_btn.click(
#             get_hint,
#             inputs=[problem_display, hint_level, current_student_id, current_attempt_id],
#             outputs=[hint_output, status]
#         )

#         evaluate_btn.click(
#             evaluate_solution,
#             inputs=[problem_display, student_code, student_explanation, current_student_id, current_session_id],
#             outputs=[evaluation_output, status, current_attempt_id]
#         )

#         get_progress_btn.click(
#             get_progress_report,
#             inputs=[progress_id],
#             outputs=[progress_report, recent_activity, status]
#         )

        # Reset hints button
        def reset_hints():
            tutor.hint_agent.hint_history = {}
            return {"message": "Hints reset"}, "🔄 Hint history reset"

        reset_hints_btn.click(
            reset_hints,
            outputs=[hint_output, status]
        )

        # Refresh agent status
        def refresh_agent_status():
            return (
                "🟢 Active - " + ("Ready to teach" if random.random() > 0.1 else "Processing..."),
                "🟢 Active - " + ("Ready to generate problems" if random.random() > 0.1 else "Generating..."),
                "🟢 Active - " + ("Ready to provide hints" if random.random() > 0.1 else "Processing..."),
                "🟢 Active - " + ("Ready to evaluate solutions" if random.random() > 0.1 else "Evaluating...")
            )

        refresh_btn.click(
            refresh_agent_status,
            outputs=[teacher_status, qgen_status, hint_status, eval_status]
        )

        # Add footer
        gr.Markdown("---")
        gr.Markdown(f"""
        <div style="text-align: center; color: #666; font-size: 0.9em;">
        <p>{APP_NAME} v{APP_VERSION} | Built with Gradio & LangChain</p>
        <p>Powered by 4 AI agents working together to teach you DSA</p>
        </div>
        """)

    return demo


# ==================== MAIN FUNCTION ====================
def main():
    """
    Main function to run the DSA Tutor application
    """
    print_config_summary()

    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)

    try:
        # Initialize database
        logger.info("Initializing database...")
        db = DatabaseManager()
        logger.info("Database initialized successfully")
        
        # Initialize the tutor system
        logger.info(f"Initializing {APP_NAME}...")
        tutor = DSATutorSystem(GROQ_API_KEY)
        logger.info("Tutor system initialized successfully")

        # Create Gradio interface with database tracking
        logger.info("Creating web interface...")
        demo = create_tutor_interface(tutor, db)

        # Launch the application
        logger.info(f"{APP_NAME} is ready!")
        logger.info(f"Launching web server on http://{SERVER_HOST}:{SERVER_PORT}")

        if SHARE_PUBLICLY:
            logger.info("Creating public share link (expires in 72 hours)...")

        demo.launch(
            server_name=SERVER_HOST,
            server_port=SERVER_PORT,
            share=SHARE_PUBLICLY,
            show_error=True,
            debug=DEBUG_MODE
        )

    except KeyboardInterrupt:
        logger.info("Shutting down DSA Tutor...")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Failed to start DSA Tutor: {e}")
        if DEBUG_MODE:
            import traceback
            traceback.print_exc()
        sys.exit(1)


# ==================== ENTRY POINT ====================
if __name__ == "__main__":
    # Check Python version
    import platform
    python_version = platform.python_version()
    print(f"Python {python_version}")

    # Import check
    try:
        import groq
        import langchain
        import gradio
        print("All required packages are installed")
    except ImportError as e:
        print(f"Missing package: {e}")
        print("Run: pip install -r requirements.txt")
        sys.exit(1)

    # Run the application
    main()