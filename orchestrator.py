# ORCHESTRATOR: DSA TUTOR SYSTEM
from agents.teacherAgent import TeacherAgent
from agents.QGenAgent import QuestionGeneratorAgent
from agents.hintAgent import HintAgent
from agents.EvalAgent import EvaluatorAgent
from knowledge_base import DSAKnowledgeBase, Topic
import gradio as gr
from typing import Dict, List
from knowledge_base import Difficulty
from datetime import datetime

class DSATutorSystem:
    """Main orchestrator for all agents"""
    
    def __init__(self, api_key: str = None):
        if api_key:
            import os
            os.environ["GROQ_API_KEY"] = api_key
        self.knowledge_base = DSAKnowledgeBase()
        self.teacher = TeacherAgent(self.knowledge_base)
        self.question_generator = QuestionGeneratorAgent(self.knowledge_base)
        self.hint_agent = HintAgent()
        self.evaluator = EvaluatorAgent()
        self.current_student = None
        self.session_history = []
        
    def start_learning_session(self, student_id: str, topics: List[str], 
                              difficulty: str = "Easy") -> Dict:
        """Start new learning session"""
        difficulty_enum = Difficulty(difficulty)
        session_info = self.teacher.start_teaching_session(
            student_id, topics, difficulty_enum
        )
        
        self.current_student = student_id
        self.session_history.append({
            "session_id": session_info["session_id"],
            "start_time": datetime.now(),
            "topics": topics,
            "difficulty": difficulty
        })
        
        return session_info
    
    def get_concept_explanation(self, concept: str, topic: str, 
                               student_level: str = "beginner") -> Dict:
        """Get explanation of a concept"""
        return self.teacher.explain_concept(concept, topic, student_level)
    
    def generate_practice_question(self, topic: str, difficulty: str = "Medium",
                                 weakness: List[str] = None) -> Dict:
        """Generate practice question"""
        difficulty_enum = Difficulty(difficulty)
        return self.question_generator.generate_question(
            topic, difficulty_enum, weakness
        )
    
    def get_hint(self, problem: Dict, hint_level: int = 0,
                student_info: Dict = None) -> Dict:
        """Get hint for problem"""
        student_code = student_info.get("code") if student_info else None
        student_approach = student_info.get("approach") if student_info else None
        
        return self.hint_agent.get_progressive_hints(
            problem, student_code, student_approach, hint_level
        )
    
    def evaluate_solution(self, problem: Dict, student_code: str,
                         explanation: str = None) -> Dict:
        """Evaluate student solution"""
        return self.evaluator.evaluate_solution(problem, student_code, explanation)
    
    def generate_learning_path(self, topics: List[str], 
                              current_skill: str = "beginner") -> List[Dict]:
        """Generate personalized learning path"""
        learning_path = []
        
        for topic in topics:
            # Start with concepts
            learning_path.append({
                "type": "concept",
                "topic": topic,
                "title": f"Introduction to {topic}",
                "duration": "15-20 minutes",
                "resources": ["Concept explanation", "Visual examples"]
            })
            
            # Add easy problem
            learning_path.append({
                "type": "problem",
                "topic": topic,
                "difficulty": "Easy",
                "title": f"Basic {topic} Practice",
                "estimated_time": "20-30 minutes"
            })
            
            # Add medium problem
            learning_path.append({
                "type": "problem",
                "topic": topic,
                "difficulty": "Medium",
                "title": f"Intermediate {topic} Challenge",
                "estimated_time": "30-45 minutes"
            })
            
            # Add review
            learning_path.append({
                "type": "review",
                "topic": topic,
                "title": f"{topic} Review & Common Patterns",
                "duration": "10-15 minutes"
            })
        
        return learning_path
    
    def get_student_progress(self, student_id: str) -> Dict:
        """Get student progress report"""
        if student_id not in self.teacher.student_progress:
            return {"error": "Student not found"}
        
        progress = self.teacher.student_progress[student_id]
        
        # Calculate overall mastery
        total_mastery = sum(progress["topics_mastery"].values())
        avg_mastery = total_mastery / len(progress["topics_mastery"]) if progress["topics_mastery"] else 0
        
        # Identify strong and weak areas
        strong_areas = [topic for topic, mastery in progress["topics_mastery"].items() 
                       if mastery >= 70]
        weak_areas = [topic for topic, mastery in progress["topics_mastery"].items() 
                     if mastery < 50]
        
        return {
            "student_id": student_id,
            "total_sessions": progress["total_sessions"],
            "average_mastery": round(avg_mastery, 1),
            "strong_areas": strong_areas,
            "weak_areas": weak_areas,
            "recommendations": self._generate_recommendations(progress)
        }
    
    def _generate_recommendations(self, progress: Dict) -> List[str]:
        """Generate personalized recommendations"""
        recommendations = []
        
        if progress["total_sessions"] < 3:
            recommendations.append("Complete more practice sessions to establish baseline")
        
        weak = progress.get("weak_areas", [])
        if weak:
            recommendations.append(f"Focus on improving: {', '.join(weak[:3])}")
        
        strong = progress.get("strong_areas", [])
        if strong:
            recommendations.append(f"Leverage your strength in {strong[0]} to tackle harder problems")
        
        return recommendations

# ================================
# GRADIO UI INTERFACE
# ================================

def create_tutor_interface():
    """Create Gradio interface for DSA Tutor"""
    
    tutor = DSATutorSystem()
    
    with gr.Blocks(title=" DSA Tutor Multi-Agent System", theme=gr.themes.Soft()) as demo:
        gr.Markdown("#  Intelligent DSA Tutor System")
        gr.Markdown("Four AI agents work together to teach you Data Structures & Algorithms")
        
        with gr.Tabs():
            # Tab 1: Learning Session
            with gr.TabItem("🎓 Learning Session"):
                with gr.Row():
                    with gr.Column(scale=1):
                        student_id = gr.Textbox(label="Student ID", value="student_001")
                        topics = gr.CheckboxGroup(
                            choices=[t.value for t in Topic],
                            label="Select Topics to Learn",
                            value=["Arrays", "Linked Lists"]
                        )
                        difficulty = gr.Dropdown(
                            choices=["Easy", "Medium", "Hard"],
                            value="Medium",
                            label="Difficulty Level"
                        )
                        start_btn = gr.Button("Start Learning Session", variant="primary")
                    
                    with gr.Column(scale=2):
                        session_output = gr.JSON(label="Session Info")
                        welcome_msg = gr.Markdown(label="Welcome Message")
                
                with gr.Row():
                    concept_input = gr.Textbox(label="Concept to Learn", placeholder="e.g., Binary Search, Dynamic Programming")
                    topic_select = gr.Dropdown(
                        choices=[t.value for t in Topic],
                        value="Arrays",
                        label="Topic"
                    )
                    explain_btn = gr.Button(" Explain Concept")
                
                with gr.Row():
                    concept_output = gr.JSON(label="Concept Explanation")
                    concept_display = gr.Markdown(label="Detailed Explanation")
            
            # Tab 2: Practice Problems
            with gr.TabItem(" Practice Problems"):
                with gr.Row():
                    with gr.Column(scale=1):
                        practice_topic = gr.Dropdown(
                            choices=[t.value for t in Topic],
                            value="Arrays",
                            label="Topic"
                        )
                        practice_difficulty = gr.Dropdown(
                            choices=["Easy", "Medium", "Hard"],
                            value="Medium",
                            label="Difficulty"
                        )
                        weaknesses = gr.Textbox(
                            label="Your Weaknesses (comma-separated)",
                            placeholder="e.g., edge cases, time complexity"
                        )
                        generate_btn = gr.Button("Generate Practice Problem")
                    
                    with gr.Column(scale=2):
                        problem_display = gr.JSON(label="Generated Problem")
                        problem_desc = gr.Markdown(label="Problem Description")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        hint_level = gr.Slider(0, 3, value=0, label="Hint Level")
                        get_hint_btn = gr.Button(" Get Hint")
                        student_code = gr.Textbox(
                            label="Your Solution Code",
                            placeholder="Paste your Python code here...",
                            lines=10
                        )
                        student_explanation = gr.Textbox(
                            label="Your Approach Explanation",
                            placeholder="Explain your approach..."
                        )
                        evaluate_btn = gr.Button( "Evaluate Solution")
                    
                    with gr.Column(scale=1):
                        hint_output = gr.JSON(label="Hint")
                        evaluation_output = gr.JSON(label="Evaluation Results")
            
            # Tab 3: Progress Tracking
            with gr.TabItem("Progress & Analytics"):
                with gr.Row():
                    progress_id = gr.Textbox(label="Student ID", value="student_001")
                    get_progress_btn = gr.Button(" Get Progress Report")
                
                with gr.Row():
                    progress_report = gr.JSON(label="Progress Report")
                    learning_path = gr.JSON(label="Recommended Learning Path")
                
                with gr.Row():
                    gr.Markdown("###  Session History")
                    session_history = gr.JSON(label="Recent Sessions")
            
            # Tab 4: Multi-Agent Dashboard
            with gr.TabItem(" Agent Dashboard"):
                gr.Markdown("## Agent Activities")
                
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("###  Teacher Agent")
                        teacher_status = gr.Textbox(label="Status", value="Active")
                        teacher_last_action = gr.Textbox(label="Last Action")
                    
                    with gr.Column():
                        gr.Markdown("###  Question Generator")
                        qgen_status = gr.Textbox(label="Status", value="Active")
                        qgen_last_question = gr.Textbox(label="Last Generated")
                    
                    with gr.Column():
                        gr.Markdown("###  Hint Agent")
                        hint_status = gr.Textbox(label="Status", value="Active")
                        hint_history = gr.Textbox(label="Hint History")
                    
                    with gr.Column():
                        gr.Markdown("###  Evaluator Agent")
                        eval_status = gr.Textbox(label="Status", value="Active")
                        eval_stats = gr.Textbox(label="Evaluation Stats")
        
        # Event Handlers
        def start_session(student_id, topics, difficulty):
            session = tutor.start_learning_session(student_id, topics, difficulty)
            return session, session.get("welcome_message", "")
        
        def explain_concept(concept, topic):
            explanation = tutor.get_concept_explanation(concept, topic)
            return explanation, explanation.get("explanation", "")
        
        def generate_problem(topic, difficulty, weaknesses):
            weakness_list = [w.strip() for w in weaknesses.split(",")] if weaknesses else []
            problem = tutor.generate_practice_question(topic, difficulty, weakness_list)
            return problem, f"### {problem.get('title', 'Problem')}\n\n{problem.get('description', '')}"
        
        def get_hint(problem, hint_level):
            if not problem:
                return {"error": "No problem provided"}
            return tutor.get_hint(problem, hint_level)
        
        def evaluate_solution(problem, code, explanation):
            if not problem or not code:
                return {"error": "Problem and code required"}
            return tutor.evaluate_solution(problem, code, explanation)
        
        def get_progress_report(student_id):
            return tutor.get_student_progress(student_id)
        
        def get_learning_path(topics):
            return tutor.generate_learning_path(topics)
        
        # Connect events
        start_btn.click(
            start_session,
            inputs=[student_id, topics, difficulty],
            outputs=[session_output, welcome_msg]
        )
        
        explain_btn.click(
            explain_concept,
            inputs=[concept_input, topic_select],
            outputs=[concept_output, concept_display]
        )
        
        generate_btn.click(
            generate_problem,
            inputs=[practice_topic, practice_difficulty, weaknesses],
            outputs=[problem_display, problem_desc]
        )
        
        get_hint_btn.click(
            get_hint,
            inputs=[problem_display, hint_level],
            outputs=[hint_output]
        )
        
        evaluate_btn.click(
            evaluate_solution,
            inputs=[problem_display, student_code, student_explanation],
            outputs=[evaluation_output]
        )
        
        get_progress_btn.click(
            get_progress_report,
            inputs=[progress_id],
            outputs=[progress_report]
        )
        
        # Initial setup
        demo.load(
            lambda: tutor.generate_learning_path(["Arrays", "Linked Lists"]),
            outputs=[learning_path]
        )
    
    return demo

# ================================
# MAIN EXECUTION
# ================================

if __name__ == "__main__":
    print(" Initializing DSA Tutor Multi-Agent System...")
    
    # Create tutor system
    tutor = DSATutorSystem()
    
    # Example usage
    print("\n1. Starting learning session...")
    session = tutor.start_learning_session(
        student_id="alice_123",
        topics=["Arrays", "Dynamic Programming"],
        difficulty="Medium"
    )
    print(f"   Session ID: {session['session_id']}")
    
    print("\n2. Getting concept explanation...")
    explanation = tutor.get_concept_explanation(
        concept="Dynamic Programming",
        topic="Dynamic Programming",
        student_level="intermediate"
    )
    print(f"   Explanation length: {len(explanation['explanation'])} chars")
    
    print("\n3. Generating practice question...")
    problem = tutor.generate_practice_question(
        topic="Arrays",
        difficulty="Medium",
        weakness=["edge cases", "optimization"]
    )
    print(f"   Problem: {problem.get('title', 'Unknown')}")
    
    print("\n4. Getting hint...")
    hint = tutor.get_hint(problem, hint_level=0)
    print(f"   Hint level {hint['hint_level']}: {hint['hint'][:100]}...")
    
    print("\n5. Evaluating sample solution...")
    sample_code = """
def twoSum(nums, target):
    hash_map = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in hash_map:
            return [hash_map[complement], i]
        hash_map[num] = i
    return []
    """
    evaluation = tutor.evaluate_solution(problem, sample_code, "Using hash map for O(n) solution")
    print(f"   Score: {evaluation['score']}/100")
    print(f"   Correct: {evaluation['correctness']}")
    
    print("\n6. Getting progress report...")
    progress = tutor.get_student_progress("alice_123")
    print(f"   Average mastery: {progress.get('average_mastery', 0)}%")
    
    print("\n DSA Tutor System is ready!")
    print("\n To launch interactive UI, run:")
    print("   demo = create_tutor_interface()")
    print("   demo.launch()") 