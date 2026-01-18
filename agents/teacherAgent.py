# AGENT 1: TEACHER AGENT
import uuid
# Defer knowledge_base import to avoid slow transformers loading
# from knowledge_base import DSAKnowledgeBase, Topic, Difficulty
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from datetime import datetime
from typing import Dict, List
import json
from utils.json_parser import extract_json

# Define Difficulty enum locally to avoid import issues
from enum import Enum
class Difficulty(Enum):
    EASY = "Easy"
    MEDIUM = "Medium" 
    HARD = "Hard"
class TeacherAgent:
    """Main teaching agent that orchestrates learning sessions"""
    
    def __init__(self, knowledge_base=None):
        self.kb = knowledge_base
        self.llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.4)
        self.student_progress = {}  # student_id --> progress = tracking progress with the help of the student ID
        self.current_session = None
        
    def start_teaching_session(self, student_id: str, topics: List[str], 
                              difficulty: Difficulty = Difficulty.EASY) -> Dict:
        """Start a new teaching session"""
        session_id = str(uuid.uuid4())
        self.current_session = {
            "session_id": session_id,
            "student_id": student_id,
            "topics": topics,
            "difficulty": difficulty,
            "start_time": datetime.now(),
            "problems_attempted": [],
            "concepts_covered": [],
            "performance_metrics": {}
        }
        
        # Initialize student progress if needed
        if student_id not in self.student_progress:
            # Import Topic locally to avoid circular/heavy imports
            from knowledge_base import Topic
            
            self.student_progress[student_id] = {
                "total_sessions": 0,
                "topics_mastery": {topic.value: 0.0 for topic in Topic},
                "weak_areas": [],
                "learning_path": []
            }
        
        self.student_progress[student_id]["total_sessions"] += 1
        
        # Generate learning plan
        learning_plan = self._generate_learning_plan(topics, difficulty)
        
        return {
            "session_id": session_id,
            "welcome_message": self._generate_welcome_message(topics, self.student_progress[student_id]),
            "learning_plan": learning_plan,
            "first_concept": learning_plan[0] if learning_plan else None
        }
    
    def _generate_learning_plan(self, topics: List[str], difficulty: Difficulty) -> List[Dict]:
        """Generate personalized 5-phase learning plan"""
        if self.kb is None:
            raise ValueError("Knowledge base not available. Please initialize the system first.")
            
        topics_str = ", ".join(topics)
        
        prompt = f"""
        Create a flexible 5-phase learning plan for a student learning: {topics_str}.
        Difficulty Level: {difficulty.value}
        
        Structure the plan into these 5 phases:
        1. Warmup / Prior Knowledge Check
        2. Concept Exploration (Core theory)
        3. Guided Practice (Walkthroughs)
        4. Independent Practice (Challenge)
        5. Reflection & Next Steps
        
        For each phase, provide:
        - Phase Name
        - Specific Topic (from the user's list)
        - Activity Type (concept, problem, question, reflection)
        - Brief Description (1 sentence)
        - Estimated Time
        
        IMPORTANT: Return ONLY valid JSON array.
        Format: [
            {{"phase": "Warmup", "topic": "...", "type": "question", "description": "...", "estimated_time": "5 mins"}},
            ...
        ]
        """
        
        response = self.llm.invoke([HumanMessage(content=prompt)])
        
        try:
            # Use robust extractor
            plan = extract_json(response.content)
            return plan
        except Exception as e:
            # Fallback simple plan if JSON fails
            return [
                {"phase": "Warmup", "topic": topics[0], "type": "concept", "description": "Quick review of basic concepts", "estimated_time": "5 mins"},
                {"phase": "Concept", "topic": topics[0], "type": "concept", "description": "Deep dive into core theory", "estimated_time": "15 mins"},
                {"phase": "Practice", "topic": topics[0], "type": "problem", "description": "Solve a practice problem", "estimated_time": "20 mins"}
            ]

    def _generate_welcome_message(self, topics: List[str], student_progress: Dict) -> str:
        """Generate highly personalized welcome message"""
        topics_str = ", ".join(topics)
        total_sessions = student_progress.get("total_sessions", 1)
        weak_areas = ", ".join(student_progress.get("weak_areas", []))
        
        context = f"This is session #{total_sessions}."
        if weak_areas:
            context += f" Focus on improving: {weak_areas}."
            
        prompt = f"""
        You are an AI DSA Tutor welcoming a student.
        Topics: {topics_str}
        Student Context: {context}
        
        Write a personalized welcome message (2-3 paragraphs) that:
        1. Welcomes them back (or first time) warm and professionally.
        2. Acknowledges their specific topics and context.
        3. Explains the session flow: Warmup -> Concept -> Guided -> Independent -> Reflection.
        4. Encourages questions: "I adapt to you. If you're stuck, just ask!"
        
        Tone: Encouraging, structured, reliable.
        """
        
        response = self.llm.invoke([HumanMessage(content=prompt)])
        return response.content
    
    def explain_concept(self, concept: str, topic: str, 
                       student_level: str = "beginner") -> Dict:
        """Explain a DSA concept at appropriate level"""
        if self.kb is None:
            raise ValueError("Knowledge base not available. Please initialize the system first.")
            
        # Retrieve relevant knowledge
        docs = self.kb.query(f"Explain {concept} in {topic}", topic)
        
        # Generate explanation
        prompt = f"""
        Explain the concept of '{concept}' in {topic} to a {student_level} student.
        
        Context from knowledge base:
        {docs[0].page_content[:1000] if docs else 'No specific context'}
        
        Requirements:
        1. Start with simple analogy or real-world example
        2. Explain key principles clearly
        3. Include time/space complexity considerations
        4. Mention common use cases
        5. Warn about common pitfalls
        6. End with a summary
        
        Make it engaging and educational. Target: 3-4 paragraphs.
        """
        
        response = self.llm.invoke([HumanMessage(content=prompt)])
        
        # Generate visual representation prompt if applicable
        visual_prompt = None
        if topic in ["Trees", "Graphs", "Linked Lists"]:
            visual_prompt = self._generate_visual_prompt(concept, topic)
        
        return {
            "explanation": response.content,
            "key_points": self._extract_key_points(response.content),
            "visual_prompt": visual_prompt,
            "related_concepts": self._get_related_concepts(topic, concept)
        }
    
    def _generate_visual_prompt(self, concept: str, topic: str) -> str:
        """Generate prompt for creating visual aid"""
        prompt = f"""
        Create a description for visualizing '{concept}' in {topic}.
        
        Describe:
        1. What elements should be in the visualization
        2. How they connect/relate
        3. Key annotations/labels needed
        4. Step-by-step progression if applicable
        
        Format as JSON with keys: elements, connections, annotations, steps.
        """
        
        response = self.llm.invoke([HumanMessage(content=prompt)])
        return response.content
    
    def _extract_key_points(self, explanation: str) -> List[str]:
        """Extract key points from explanation"""
        prompt = f"""
        Extract 5-7 key points from this explanation:
        
        {explanation}
        
        Return as a JSON list of strings.
        """
        
        response = self.llm.invoke([HumanMessage(content=prompt)])
        try:
            return json.loads(response.content)
        except:
            # Fallback to simple extraction
            lines = explanation.split('\n')
            return [line.strip() for line in lines if line.strip() and len(line.strip()) > 20][:5]
    
    def _get_related_concepts(self, topic: str, concept: str) -> List[str]:
        """Get related concepts for deeper learning"""
        if self.kb is None:
            return []  # Return empty list if kb not available
            
        docs = self.kb.query(f"Concepts related to {concept} in {topic}", topic)
        
        prompt = f"""
        Based on this context about {topic}, list 3-5 closely related concepts to '{concept}':
        
        Context: {docs[0].page_content[:500] if docs else ''}
        
        Return as JSON list of strings.
        """
        
        response = self.llm.invoke([HumanMessage(content=prompt)])
        try:
            return json.loads(response.content)
        except:
            return []