# ================================
# AGENT 2: QUESTION GENERATOR AGENT
# ================================
from knowledge_base import DSAKnowledgeBase, Difficulty
from langchain_groq import ChatGroq
from typing import Dict, List
import uuid
from datetime import datetime
import json
# from langchain.schema import HumanMessage
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage


class QuestionGeneratorAgent:
    """Generates personalized practice questions"""
    
    def __init__(self, knowledge_base: DSAKnowledgeBase):
        self.kb = knowledge_base
        self.llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.8)
        self.problem_templates = self._load_problem_templates()
    
    def _load_problem_templates(self) -> Dict:
        """Load templates for different types of problems"""
        return {
            "arrays": {
                "patterns": ["two-pointer", "sliding-window", "prefix-sum", "rotation"],
                "difficulty_factors": {
                    "Easy": ["single pattern", "small input", "clear constraints"],
                    "Medium": ["combined patterns", "edge cases", "optimization needed"],
                    "Hard": ["multiple patterns", "large input", "complex constraints"]
                }
            },
            "linked_lists": {
                "patterns": ["reversal", "cycle detection", "merge", "two-pointer"],
                "difficulty_factors": {
                    "Easy": ["single operation", "no edge cases"],
                    "Medium": ["multiple operations", "special cases"],
                    "Hard": ["complex manipulation", "memory constraints"]
                }
            },
            "trees": {
                "patterns": ["traversal", "bst validation", "lca", "path sum"],
                "difficulty_factors": {
                    "Easy": ["basic traversal", "single property"],
                    "Medium": ["combined properties", "modified traversal"],
                    "Hard": ["multiple trees", "complex constraints"]
                }
            }
        }
    
    def generate_question(self, topic: str, difficulty: Difficulty,
                         student_weakness: List[str] = None) -> Dict:
        """Generate a new practice question"""
        # Get relevant knowledge
        docs = self.kb.query(f"{difficulty.value} problem about {topic}", topic)
        
        # Generate question using LLM
        weakness_context = ""
        if student_weakness:
            weakness_context = f"\nFocus on these student weaknesses: {', '.join(student_weakness)}"
        
        prompt = f"""
        Generate a {difficulty.value} difficulty DSA problem about {topic}.
        
        Context from knowledge base:
        {docs[0].page_content[:1000] if docs else 'General DSA problem'}
        
        Requirements:
        1. Create a clear, unambiguous problem statement
        2. Include input/output format
        3. Provide 2-3 sample test cases with explanations
        4. Specify constraints
        5. Make it appropriate for {difficulty.value} level
        {weakness_context}
        
        Format the response as JSON with keys:
        - title: Problem title
        - description: Detailed problem statement
        - input_format: How input is provided
        - output_format: Expected output
        - constraints: Time/space limits, input ranges
        - examples: List of example inputs/outputs with explanations
        - hints: 2-3 hints for solving
        """
        
        response = self.llm.invoke([HumanMessage(content=prompt)])
        
        try:
            problem = json.loads(response.content)
            
            # Add metadata
            problem["topic"] = topic
            problem["difficulty"] = difficulty.value
            problem["generated_at"] = datetime.now().isoformat()
            problem["id"] = f"gen_{hash(str(problem)) % 10000:04d}"
            
            return problem
        except json.JSONDecodeError:
            # Fallback to structured generation
            return self._create_fallback_problem(topic, difficulty, response.content)
    
    def _create_fallback_problem(self, topic: str, difficulty: Difficulty, 
                                content: str) -> Dict:
        """Create problem from unstructured LLM output"""
        return {
            "id": f"fallback_{str(uuid.uuid4())[:8]}",
            "title": f"{topic} Problem",
            "description": content[:500],
            "topic": topic,
            "difficulty": difficulty.value,
            "input_format": "Standard input format",
            "output_format": "Standard output format",
            "constraints": "1 <= n <= 10^5",
            "examples": [{"input": "Sample input", "output": "Sample output"}],
            "hints": ["Think about the core concept", "Consider edge cases"],
            "generated_at": datetime.now().isoformat()
        }
    
    def generate_multiple_choice(self, concept: str, topic: str, 
                               num_options: int = 4) -> Dict:
        """Generate multiple choice questions about concepts"""
        prompt = f"""
        Generate a multiple choice question about '{concept}' in {topic}.
        
        Requirements:
        1. Create a clear question stem
        2. Generate {num_options} options (A, B, C, D)
        3. Exactly one correct answer
        4. Include explanations for why each option is correct/incorrect
        5. Make it test conceptual understanding, not just memorization
        
        Format as JSON with keys: question, options (list), correct_answer (index), explanations (list).
        """
        
        response = self.llm.invoke([HumanMessage(content=prompt)])
        
        try:
            mcq = json.loads(response.content)
            mcq["concept"] = concept
            mcq["topic"] = topic
            return mcq
        except:
            return self._create_fallback_mcq(concept, topic)
    
    def _create_fallback_mcq(self, concept: str, topic: str) -> Dict:
        """Create fallback MCQ"""
        return {
            "question": f"What is the key characteristic of {concept} in {topic}?",
            "options": [
                "Option A - Basic property",
                "Option B - Advanced property",
                "Option C - Common misconception",
                "Option D - Correct characteristic"
            ],
            "correct_answer": 3,
            "explanations": [
                "A is too basic",
                "B is incorrect",
                "C is wrong",
                "D correctly describes the concept"
            ],
            "concept": concept,
            "topic": topic
        }
    
    def generate_variations(self, base_problem: Dict, num_variations: int = 3) -> List[Dict]:
        """Generate variations of a problem (easier/harder)"""
        variations = []
        
        for i in range(num_variations):
            # Alternate between easier and harder
            if i % 2 == 0:
                difficulty = "Easier variation: Simplify constraints or requirements"
            else:
                difficulty = "Harder variation: Add constraints or requirements"
            
            prompt = f"""
            Create a variation of this problem:
            
            Original: {base_problem['description'][:500]}
            
            Create a {difficulty}.
            
            Requirements:
            1. Keep the core concept the same
            2. Modify constraints or requirements significantly
            3. Provide new sample test cases
            4. Update hints if needed
            
            Format as JSON with same structure as original.
            """
            
            response = self.llm.invoke([HumanMessage(content=prompt)])
            
            try:
                variation = json.loads(response.content)
                variation["original_id"] = base_problem["id"]
                variation["is_variation"] = True
                variation["variation_type"] = "easier" if i % 2 == 0 else "harder"
                variations.append(variation)
            except:
                continue
        
        return variations