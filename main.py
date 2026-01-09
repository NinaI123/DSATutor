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
def create_tutor_interface(tutor: DSATutorSystem):
    """
    Create the Gradio web interface for the DSA Tutor
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
        
        with gr.Tabs():
            # ========== TAB 1: LEARNING SESSION ==========
            with gr.TabItem("🎓 Learning Session", id=1):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### Student Profile")
                        student_id = gr.Textbox(
                            label="Student ID",
                            value="student_001",
                            info="Enter a unique ID for tracking progress"
                        )
                        
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
                            "Start Learning Session",
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
                    explain_btn = gr.Button("Explain Concept", variant="secondary")
                
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
                            reset_hints_btn = gr.Button(" Reset Hints")
                        
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
                        
                        evaluate_btn = gr.Button(" Evaluate Solution", variant="primary")
                    
                    with gr.Column(scale=1):
                        gr.Markdown("### Feedback")
                        hint_output = gr.JSON(label="Hint")
                        evaluation_output = gr.JSON(label="Evaluation Results")
            
            # ========== TAB 3: PROGRESS TRACKING ==========
            with gr.TabItem("Progress & Analytics", id=3):
                with gr.Row():
                    progress_id = gr.Textbox(
                        label="Student ID",
                        value="student_001",
                        info="Enter student ID to view progress"
                    )
                    get_progress_btn = gr.Button("Get Progress Report", variant="primary")
                
                with gr.Row():
                    progress_report = gr.JSON(label="Progress Report")
                    learning_path = gr.JSON(label="Recommended Learning Path")
                
                with gr.Row():
                    gr.Markdown("### 📋 Session History")
                    session_history = gr.JSON(label="Recent Sessions")
            
            # ========== TAB 4: AGENT DASHBOARD ==========
            with gr.TabItem("Agent Dashboard", id=4):
                gr.Markdown("## Multi-Agent System Status")
                
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Teacher Agent")
                        teacher_status = gr.Textbox(
                            label="Status",
                            value="Active - Ready to teach",
                            interactive=False
                        )
                        teacher_stats = gr.Textbox(
                            label="Statistics",
                            value="Concepts explained: 0\nSessions conducted: 0",
                            interactive=False
                        )
                    
                    with gr.Column():
                        gr.Markdown("### Question Generator")
                        qgen_status = gr.Textbox(
                            label="Status", 
                            value=" Active - Ready to generate problems",
                            interactive=False
                        )
                        qgen_stats = gr.Textbox(
                            label="Statistics",
                            value="Problems generated: 0\nDifficulty distribution: N/A",
                            interactive=False
                        )
                
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("###  Hint Agent")
                        hint_status = gr.Textbox(
                            label="Status",
                            value=" Active - Ready to provide hints",
                            interactive=False
                        )
                        hint_stats = gr.Textbox(
                            label="Statistics",
                            value="Hints given: 0\nAverage hint level: 0.0",
                            interactive=False
                        )
                    
                    with gr.Column():
                        gr.Markdown("###  Evaluator Agent")
                        eval_status = gr.Textbox(
                            label="Status",
                            value="Active - Ready to evaluate solutions",
                            interactive=False
                        )
                        eval_stats = gr.Textbox(
                            label="Statistics",
                            value="Solutions evaluated: 0\nAverage score: 0.0",
                            interactive=False
                        )
                
                with gr.Row():
                    refresh_btn = gr.Button(" Refresh Agent Status")
        
        # ==================== EVENT HANDLERS ====================
        def start_session(student_id, topics, difficulty):
            try:
                session = tutor.start_learning_session(student_id, topics, difficulty)
                return session, session.get("welcome_message", ""), " Session started successfully"
            except Exception as e:
                logging.error(f"Error starting session: {e}")
                return {}, f"Error starting session: {str(e)}", f"Error: {str(e)}"
        
        def explain_concept(concept, topic):
            if not concept.strip():
                return {}, "Please enter a concept to learn", " Please enter a concept to learn"
            try:
                explanation = tutor.get_concept_explanation(concept, topic)
                return explanation, explanation.get("explanation", ""), f" Explained '{concept}' in {topic}"
            except Exception as e:
                logging.error(f"Error explaining concept: {e}")
                return {}, f"Error explaining concept: {str(e)}", f" Error: {str(e)}"
        
        def generate_problem(topic, difficulty, weaknesses):
            try:
                weakness_list = [w.strip() for w in weaknesses.split(",")] if weaknesses else []
                problem = tutor.generate_practice_question(topic, difficulty, weakness_list)
                return problem, f"### {problem.get('title', 'Problem')}\n\n{problem.get('description', '')}", f"🎲 Generated {difficulty} problem on {topic}"
            except Exception as e:
                logging.error(f"Error generating problem: {e}")
                return {}, f"Error generating problem: {str(e)}", f" Error: {str(e)}"
        
        def get_hint(problem, hint_level):
            if not problem:
                return {"error": "No problem provided"}, " No problem provided"
            try:
                hint = tutor.get_hint(problem, hint_level)
                return hint, f" Hint level {hint['hint_level']} provided"
            except Exception as e:
                logging.error(f"Error getting hint: {e}")
                return {"error": str(e)}, f" Error: {str(e)}"
        
        def evaluate_solution(problem, code, explanation):
            if not problem or not code:
                return {"error": "Problem and code required"}, " Problem and code required"
            try:
                evaluation = tutor.evaluate_solution(problem, code, explanation)
                score = evaluation.get('score', 0)
                return evaluation, f" Solution evaluated: {score}/100"
            except Exception as e:
                logging.error(f"Error evaluating solution: {e}")
                return {"error": str(e)}, f"Error: {str(e)}"
        
        def get_progress_report(student_id):
            try:
                progress = tutor.get_student_progress(student_id)
                return progress, f" Progress report generated for {student_id}"
            except Exception as e:
                logging.error(f"Error getting progress: {e}")
                return {"error": str(e)}, f" Error: {str(e)}"
        
        # Connect event handlers
        start_btn.click(
            start_session,
            inputs=[student_id, topics, difficulty],
            outputs=[session_output, welcome_msg, status]
        )
        
        explain_btn.click(
            explain_concept,
            inputs=[concept_input, topic_select],
            outputs=[concept_output, concept_display, status]
        )
        
        generate_btn.click(
            generate_problem,
            inputs=[practice_topic, practice_difficulty, weaknesses],
            outputs=[problem_display, problem_desc, status]
        )
        
        get_hint_btn.click(
            get_hint,
            inputs=[problem_display, hint_level],
            outputs=[hint_output, status]
        )
        
        evaluate_btn.click(
            evaluate_solution,
            inputs=[problem_display, student_code, student_explanation],
            outputs=[evaluation_output, status]
        )
        
        get_progress_btn.click(
            get_progress_report,
            inputs=[progress_id],
            outputs=[progress_report, status]
        )
        
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
                " Active - " + ("Ready to teach" if random.random() > 0.1 else "Processing..."),
                " Active - " + ("Ready to generate problems" if random.random() > 0.1 else "Generating..."),
                " Active - " + ("Ready to provide hints" if random.random() > 0.1 else "Processing..."),
                " Active - " + ("Ready to evaluate solutions" if random.random() > 0.1 else "Evaluating...")
            )
        
        refresh_btn.click(
            refresh_agent_status,
            outputs=[teacher_status, qgen_status, hint_status, eval_status]
        )
        
        # Add footer
        gr.Markdown("---")
        gr.Markdown(f"""
        <div style="text-align: center; color: #666; font-size: 0.9em;">
        <p>{APP_NAME} v{APP_VERSION} | Built with  using LangChain & Groq AI</p>
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
        # Initialize the tutor system
        logger.info(f" Initializing {APP_NAME}...")
        tutor = DSATutorSystem(GROQ_API_KEY)
        logger.info(" Tutor system initialized successfully")
        
        # Create Gradio interface
        logger.info(" Creating web interface...")
        demo = create_tutor_interface(tutor)
        
        # Launch the application
        logger.info(f" {APP_NAME} is ready!")
        logger.info(f" Launching web server on http://{SERVER_HOST}:{SERVER_PORT}")
        
        if SHARE_PUBLICLY:
            logger.info("🔗 Creating public share link (expires in 72 hours)...")
        
        demo.launch(
            server_name=SERVER_HOST,
            server_port=SERVER_PORT,
            # share=SHARE_PUBLICLY,
            show_error=True,
            debug=DEBUG_MODE,
            share=True
        )
        
    except KeyboardInterrupt:
        logger.info(" Shutting down DSA Tutor...")
        sys.exit(0)
    except Exception as e:
        logger.error(f" Failed to start DSA Tutor: {e}")
        if DEBUG_MODE:
            import traceback
            traceback.print_exc()
        sys.exit(1)

# ==================== ENTRY POINT ====================
if __name__ == "__main__":
    # Check Python version
    import platform
    python_version = platform.python_version()
    print(f" Python {python_version}")
    
    # Import check
    try:
        import groq
        import langchain
        import gradio
        print(" All required packages are installed")
    except ImportError as e:
        print(f" Missing package: {e}")
        print("Run: pip install -r requirements.txt")
        sys.exit(1)
    
    # Run the application
    main()