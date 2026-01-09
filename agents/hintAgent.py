# ================================
# AGENT 3: HINT AGENT
# ================================
from typing import Dict, List
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
class HintAgent:
    """Provides intelligent, progressive hints"""
    
    def __init__(self):
        self.llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.3)
        self.hint_history = {}  # problem_id -> hint_level
    
    def get_progressive_hints(self, problem: Dict, student_code: str = None,
                            student_approach: str = None, hint_level: int = 0) -> Dict:
        """Get hints at appropriate level based on student progress"""
        problem_id = problem.get("id", "unknown")
        
        # Determine hint level
        if problem_id not in self.hint_history:
            self.hint_history[problem_id] = 0
        
        current_level = max(self.hint_history[problem_id], hint_level)
        
        # Generate hint based on level
        if current_level == 0:
            hint = self._get_level_0_hint(problem)
        elif current_level == 1:
            hint = self._get_level_1_hint(problem, student_approach)
        elif current_level == 2:
            hint = self._get_level_2_hint(problem, student_code)
        else:
            hint = self._get_level_3_hint(problem)
        
        # Update history
        self.hint_history[problem_id] = current_level + 1
        
        return {
            "hint": hint,
            "hint_level": current_level,
            "max_level": 3,
            "next_level_available": current_level < 3
        }
    
    def _get_level_0_hint(self, problem: Dict) -> str:
        """General conceptual hint"""
        prompt = f"""
        Problem: {problem.get('title')}
        Description: {problem.get('description', '')[:500]}
        
        Provide a gentle, conceptual hint that:
        1. Points to the right approach without giving it away
        2. Mentions the key DSA concept needed
        3. Suggests what to think about
        4. Encourages the student
        
        Make it 2-3 sentences, friendly tone.
        """
        
        response = self.llm.invoke([HumanMessage(content=prompt)])
        return response.content
    
    def _get_level_1_hint(self, problem: Dict, student_approach: str = None) -> str:
        """More specific hint based on student's approach"""
        approach_context = f"\nStudent mentioned they're trying: {student_approach}" if student_approach else ""
        
        prompt = f"""
        Problem: {problem.get('title')}
        Description: {problem.get('description', '')[:500]}
        {approach_context}
        
        Provide a more specific hint that:
        1. Gently corrects misconceptions if any
        2. Points to a specific aspect of the problem
        3. Suggests a subproblem to solve first
        4. Mentions time/space complexity considerations
        
        Make it 3-4 sentences, instructional but not giving away solution.
        """
        
        response = self.llm.invoke([HumanMessage(content=prompt)])
        return response.content
    
    def _get_level_2_hint(self, problem: Dict, student_code: str = None) -> str:
        """Hint based on student's code"""
        code_context = f"\nStudent's current code:\n{student_code}" if student_code else ""
        
        prompt = f"""
        Problem: {problem.get('title')}
        Description: {problem.get('description', '')[:500]}
        {code_context}
        
        Provide a specific implementation hint that:
        1. Points to a specific bug or inefficiency (without directly saying "bug")
        2. Suggests a specific data structure or algorithm part
        3. Mentions edge cases to consider
        4. Gives a small code snippet idea if helpful
        
        Make it specific but still making the student think.
        """
        
        response = self.llm.invoke([HumanMessage(content=prompt)])
        return response.content
    
    def _get_level_3_hint(self, problem: Dict) -> str:
        """Near-solution hint"""
        prompt = f"""
        Problem: {problem.get('title')}
        Description: {problem.get('description', '')[:500]}
        
        Provide a strong hint that almost gives the solution:
        1. Describe the algorithm steps in general terms
        2. Mention the exact data structures to use
        3. Give the time/space complexity
        4. Still encourage the student to implement it themselves
        
        Make it clear but still require implementation work.
        """
        
        response = self.llm.invoke([HumanMessage(content=prompt)])
        return response.content
    
    def get_socratic_hint(self, problem: Dict, student_question: str) -> str:
        """Answer student questions with Socratic method (questions back)"""
        prompt = f"""
        Problem: {problem.get('title')}
        Student asks: {student_question}
        
        Respond using Socratic method:
        1. Don't give direct answer
        2. Ask guiding questions back
        3. Help student think through the problem
        4. Point to relevant concepts
        
        Ask 2-3 thoughtful questions that guide the student.
        """
        
        response = self.llm.invoke([HumanMessage(content=prompt)])
        return response.content
    
    def analyze_stuck_point(self, problem: Dict, student_actions: List[str]) -> Dict:
        """Analyze where student is stuck and provide targeted help"""
        prompt = f"""
        Problem: {problem.get('title')}
        Student actions (in order):
        {chr(10).join(f'- {action}' for action in student_actions)}
        
        Analyze:
        1. Where is the student likely stuck?
        2. What misconception might they have?
        3. What's the next conceptual step they need?
        
        Provide analysis and one specific suggestion.
        """
        
        response = self.llm.invoke([HumanMessage(content=prompt)])
        
        return {
            "analysis": response.content,
            "suggestion": self._extract_suggestion(response.content),
            "likely_stuck_point": self._identify_stuck_point(student_actions)
        }
    
    def _extract_suggestion(self, analysis: str) -> str:
        """Extract concrete suggestion from analysis"""
        lines = analysis.split('\n')
        for line in lines:
            if 'suggestion' in line.lower() or 'try' in line.lower() or 'should' in line.lower():
                return line.strip()
        return "Review the problem constraints and consider edge cases."
    
    def _identify_stuck_point(self, actions: List[str]) -> str:
        """Identify common stuck points"""
        action_text = ' '.join(actions).lower()
        
        if any(word in action_text for word in ['input', 'read', 'parse']):
            return "Understanding input format"
        elif any(word in action_text for word in ['output', 'print', 'return']):
            return "Understanding output format"
        elif any(word in action_text for word in ['loop', 'iterate', 'for', 'while']):
            return "Implementing iteration logic"
        elif any(word in action_text for word in ['condition', 'if', 'else']):
            return "Handling edge cases"
        elif any(word in action_text for word in ['data structure', 'list', 'dict', 'set']):
            return "Choosing appropriate data structure"
        else:
            return "Understanding problem requirements"