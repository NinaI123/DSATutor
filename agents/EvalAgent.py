# ================================
# AGENT 4: EVALUATOR AGENT
# ================================
from knowledge_base import DSAKnowledgeBase, Difficulty
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from typing import Dict, List
import json
import logging
from utils.json_parser import extract_json

logger = logging.getLogger(__name__)
class EvaluatorAgent:
    """Evaluates student solutions and provides feedback"""
    
    def __init__(self):
        self.llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.4)
        # Simple in-memory cache for LLM responses
        self._response_cache = {}
        self._cache_max_size = 50
    
    def _get_cached_response(self, cache_key: str, prompt: str):
        """Get cached response or make new LLM call"""
        if cache_key in self._response_cache:
            return self._response_cache[cache_key]
        
        response = self.llm.invoke([HumanMessage(content=prompt)])
        
        # Cache the response
        if len(self._response_cache) < self._cache_max_size:
            self._response_cache[cache_key] = response
        
        return response
    
    def evaluate_solution(self, problem: Dict, student_code: str,
                         student_explanation: str = None) -> Dict:
        """Comprehensive evaluation of student solution - optimized single LLM call"""
        
        # Run basic checks (fast, no API call)
        syntax_check = self._check_syntax(student_code)
        
        # Single comprehensive LLM evaluation
        evaluation_result = self._comprehensive_evaluation(problem, student_code, student_explanation)
        
        # Extract components from the single response
        test_results = evaluation_result.get("test_results", {})
        feedback = evaluation_result.get("feedback", {})
        improvements = evaluation_result.get("improvements", [])
        
        # Calculate score
        score = self._calculate_score(syntax_check, test_results, feedback)
        
        return {
            "score": score,
            "syntax_check": syntax_check,
            "test_results": test_results,
            "feedback": feedback,
            "improvements": improvements,
            "correctness": score >= 70,
            "next_steps": self._recommend_next_steps(score, feedback)
        }
    
    def _comprehensive_evaluation(self, problem: Dict, code: str, explanation: str = None) -> Dict:
        """Single comprehensive LLM evaluation combining all analysis"""
        prompt = f"""
        Analyze this student's solution comprehensively. Return ONLY valid JSON.

        Problem: {problem.get('title')}
        Description: {problem.get('description', '')}
        
        Sample test cases: {json.dumps(problem.get('examples', []), indent=2)}
        
        Student Code:
        ```python
        {code}
        ```
        
        Student Explanation: {explanation or 'No explanation provided'}
        
        Provide a complete evaluation in this exact JSON format:
        {{
          "reasoning": "Briefly explain your thought process here (this helps accuracy)",
          "test_results": {{
            "approach_correct": true/false,
            "edge_cases_handled": ["case1", "case2"],
            "time_complexity": "O(n)",
            "space_complexity": "O(1)",
            "potential_bugs": ["bug1", "bug2"]
          }},
          "feedback": {{
            "positives": ["positive1", "positive2"],
            "improvements_needed": ["improvement1", "improvement2"],
            "concept_gaps": ["gap1", "gap2"],
            "specific_suggestions": ["suggestion1", "suggestion2"]
          }},
          "improvements": [
            {{
              "type": "code",
              "change": "what to change",
              "reason": "why to change",
              "example": "code example"
            }}
          ]
        }}
        
        CRITICAL: 
        1. Parse the student code carefully.
        2. Be specific, not generic. Cite line numbers if possible.
        3. Return ONLY valid JSON.
        4. Do NOT use newlines/control characters inside JSON strings. Use \\n for line breaks.
        """
        
        # Create cache key based on problem and code (ignore explanation for caching)
        cache_key = f"{hash(problem.get('title', '') + code[:100])}"
        response = self._get_cached_response(cache_key, prompt)
        
        try:
            return extract_json(response.content)
        except Exception as e:
            logger.error(f"Failed to parse comprehensive evaluation: {e}")
            logger.debug(f"Raw response: {response.content[:300]}")
            return {
                "test_results": {
                    "approach_correct": False,
                    "edge_cases_handled": [],
                    "time_complexity": "Unknown",
                    "space_complexity": "Unknown",
                    "potential_bugs": ["Could not parse evaluation"]
                },
                "feedback": {
                    "positives": ["Attempted to solve the problem"],
                    "improvements_needed": ["Code needs more work"],
                    "concept_gaps": [],
                    "specific_suggestions": ["Review the problem requirements"]
                },
                "improvements": []
            }
    
    def _check_syntax(self, code: str) -> Dict:
        """Basic syntax and style checking"""
        issues = []
        
        # Simple pattern checks (in production, use AST or linter)
        if len(code.strip()) < 10:
            issues.append("Code appears incomplete")
        
        if 'def ' not in code and 'class ' not in code:
            issues.append("No function or class definition found")
        
        # Check for common issues
        if 'while True:' in code and 'break' not in code:
            issues.append("Potential infinite loop detected")
        
        if 'import ' in code:
            issues.append("Note: Imports detected - ensure they're allowed")
        
        return {
            "has_issues": len(issues) > 0,
            "issues": issues,
            "code_length": len(code),
            "line_count": len(code.split('\n'))
        }
    
    def _run_conceptual_tests(self, problem: Dict, code: str) -> Dict:
        """Conceptual test evaluation using LLM"""
        prompt = f"""
        Problem: {problem.get('title')}
        Description: {problem.get('description', '')}
        
        Sample test cases from problem:
        {json.dumps(problem.get('examples', []), indent=2)}
        
        Student's code:
        ```python
        {code}
        ```
        
        Evaluate conceptually:
        1. Does the code implement the right approach?
        2. What edge cases does it handle/miss?
        3. What's the time/space complexity?
        4. Any obvious bugs or inefficiencies?
        
        IMPORTANT: Return ONLY valid JSON with no markdown, no explanations, no extra text.
        Format: {{"approach_correct": true/false, "edge_cases_handled": [...], "time_complexity": "O(n)", "space_complexity": "O(1)", "potential_bugs": [...]}}
        """
        
        response = self.llm.invoke([HumanMessage(content=prompt)])
        
        try:
            return extract_json(response.content)
        except Exception as e:
            logger.error(f"Failed to parse test results: {e}")
            logger.debug(f"Raw response: {response.content[:300]}")
            return {
                "approach_correct": False,
                "edge_cases_handled": [],
                "time_complexity": "Unknown",
                "space_complexity": "Unknown",
                "potential_bugs": ["Could not parse evaluation"]
            }
    
    def _generate_feedback(self, problem: Dict, code: str, 
                          explanation: str, test_results: Dict) -> Dict:
        """Generate detailed feedback"""
        prompt = f"""
        Problem: {problem.get('title')}
        Student Code:
        ```python
        {code}
        ```
        
        Student Explanation: {explanation or 'No explanation provided'}
        
        Test Results:
        {json.dumps(test_results, indent=2)}
        
        Generate constructive feedback that:
        1. Starts with positive reinforcement
        2. Points out specific strengths
        3. Identifies specific areas for improvement
        4. Explains concepts that were misunderstood
        5. Suggests concrete next steps
        
        IMPORTANT: Return ONLY valid JSON with no markdown, no explanations, no extra text.
        Format: {{"positives": [...], "improvements_needed": [...], "concept_gaps": [...], "specific_suggestions": [...]}}
        """
        
        response = self.llm.invoke([HumanMessage(content=prompt)])
        
        try:
            return extract_json(response.content)
        except Exception as e:
            logger.error(f"Failed to parse feedback: {e}")
            logger.debug(f"Raw response: {response.content[:300]}")
            return {
                "positives": ["Attempted to solve the problem"],
                "improvements_needed": ["Code needs more work"],
                "concept_gaps": [],
                "specific_suggestions": ["Review the problem requirements"]
            }
    
    def _calculate_score(self, syntax_check: Dict, test_results: Dict, 
                        feedback: Dict) -> float:
        """Calculate overall score (0-100)"""
        score = 0.0
        
        # Syntax (20%)
        if not syntax_check["has_issues"]:
            score += 20
        
        # Approach correctness (40%)
        if test_results.get("approach_correct", False):
            score += 40
        
        # Edge cases (20%)
        edge_cases = test_results.get("edge_cases_handled", [])
        if len(edge_cases) >= 2:  # Handles multiple edge cases
            score += 20
        elif len(edge_cases) == 1:
            score += 10
        
        # Code quality (20%) - based on feedback
        if len(feedback.get("positives", [])) >= 2:
            score += 20
        elif len(feedback.get("positives", [])) == 1:
            score += 10
        
        return min(100, score)
    
    def _suggest_improvements(self, code: str, problem: Dict, 
                             feedback: Dict) -> List[Dict]:
        """Suggest specific improvements"""
        suggestions = []
        
        # Based on feedback
        for improvement in feedback.get("improvements_needed", []):
            suggestions.append({
                "type": "general",
                "suggestion": improvement,
                "priority": "high" if "critical" in improvement.lower() else "medium"
            })
        
        # Code-specific suggestions
        prompt = f"""
        Code to improve:
        ```python
        {code}
        ```
        
        Problem: {problem.get('title')}
        
        Suggest 2-3 specific code improvements with:
        1. What to change
        2. Why to change it
        3. Example of better code
        
        IMPORTANT: Return ONLY valid JSON array with no markdown, no explanations, no extra text.
        Format: [{{"change": "...", "reason": "...", "example": "..."}}, ...]
        """
        
        response = self.llm.invoke([HumanMessage(content=prompt)])
        
        try:
            code_suggestions = extract_json(response.content)
            suggestions.extend([
                {"type": "code", **sugg} for sugg in code_suggestions
            ])
        except Exception as e:
            logger.debug(f"Could not parse code suggestions: {e}")
        
        return suggestions[:5]  # Limit to 5 suggestions
    
    def _recommend_next_steps(self, score: float, feedback: Dict) -> List[str]:
        """Recommend next learning steps based on performance"""
        if score >= 90:
            return [
                "Try a harder variation of this problem",
                "Explain your solution to reinforce learning",
                "Help another student with similar problem"
            ]
        elif score >= 70:
            return [
                "Review the suggested improvements",
                "Try the problem again with optimizations",
                "Practice similar problems"
            ]
        elif score >= 50:
            return [
                "Review the core concept",
                "Study the solution approach",
                "Try a simpler version first"
            ]
        else:
            return [
                "Review fundamental concepts",
                "Start with easier problems",
                "Ask for more hints on this problem"
            ]
    
    def compare_with_optimal(self, student_code: str, optimal_solution: str) -> Dict:
        """Compare student solution with optimal solution"""
        prompt = f"""
        Student Solution:
        ```python
        {student_code}
        ```
        
        Optimal Solution:
        ```python
        {optimal_solution}
        ```
        
        Compare and provide analysis:
        1. Key differences in approach
        2. Efficiency comparison
        3. Readability comparison
        4. What student can learn from optimal solution
        
        Return as JSON with: differences (list), efficiency_analysis (str), 
        readability_comparison (str), key_learnings (list).
        """
        
        response = self.llm.invoke([HumanMessage(content=prompt)])
        
        try:
            return extract_json(response.content)
        except Exception as e:
            logger.error(f"Failed to parse comparison: {e}")
            return {
                "differences": ["Could not parse comparison"],
                "efficiency_analysis": "Unknown",
                "readability_comparison": "Unknown",
                "key_learnings": []
            }