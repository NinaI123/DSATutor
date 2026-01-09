# INSTALLATION & IMPORTS
# !pip install langchain langchain-google-genai faiss-cpu tiktoken chromadb pydantic
# !pip install gradio plotly networkx matplotlib
# !pip install python-Levenshtein  # For similarity checking

import gradio as gr
import json
import uuid
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
import random
import re
import os

# LangChain imports
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS, Chroma
from langchain_huggingface import HuggingFaceEmbeddings


from pydantic import BaseModel, Field

# DATA STRUCTURES & KNOWLEDGE BASE

class Difficulty(Enum):
    EASY = "Easy"
    MEDIUM = "Medium"
    HARD = "Hard"

class Topic(Enum):
    ARRAYS = "Arrays"
    LINKED_LISTS = "Linked Lists"
    TREES = "Trees"
    GRAPHS = "Graphs"
    SORTING = "Sorting"
    SEARCHING = "Searching"
    DYNAMIC_PROGRAMMING = "Dynamic Programming"
    RECURSION = "Recursion"
    BACKTRACKING = "Backtracking"
    QUEUES = "Queues"
    STACKS = "Stacks"


@dataclass
class DSAProblem:
    id: str
    title: str
    description: str
    topic: Topic
    difficulty: Difficulty
    code_template: str
    test_cases: List[Dict]
    optimal_solution: str
    time_complexity: str
    space_complexity: str
    hints: List[str]
    explanation: str
    similar_problems: List[str]

class DSAKnowledgeBase:
    """RAG-based knowledge base for DSA concepts - RAG docs"""
    
    def __init__(self):
        self.documents = self._load_dsa_documents()
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.vector_store = self._create_vector_store()
        
    def _load_dsa_documents(self) -> List[Document]:
        """Load comprehensive DSA knowledge"""
        dsa_topics = {
            "Arrays": """
            Arrays are contiguous memory locations storing elements of the same type.
            
            Key Concepts:
            1. Indexing: O(1) access
            2. Insertion/Deletion: O(n) worst-case
            3. Memory: Contiguous, fixed size
            4. Common operations: Traversal, search, rotation
            
            Important Algorithms:
            - Two Pointer Technique
            - Sliding Window
            - Prefix Sum
            - Dutch National Flag
            
            Common Problems:
            - Maximum Subarray Sum (Kadane's Algorithm)
            - Rotate Array
            - Merge Sorted Arrays
            - Find Missing Number
            """,
            
            "Linked Lists": """
            Linked Lists are linear data structures with nodes containing data and pointer to next node.
            
            Types:
            1. Singly Linked List
            2. Doubly Linked List
            3. Circular Linked List
            
            Operations Complexity:
            - Access: O(n)
            - Insertion/Deletion at head: O(1)
            - Insertion/Deletion at tail: O(n) without tail pointer
            
            Important Techniques:
            - Fast & Slow Pointers
            - Reverse Linked List
            - Merge Two Sorted Lists
            - Detect Cycle
            
            Common Patterns:
            - Dummy Head Pattern
            - In-place Reversal
            - Two Pointers
            """,
            
            "Trees": """
            Trees are hierarchical data structures with root node and children.
            
            Binary Tree Types:
            1. Full Binary Tree
            2. Complete Binary Tree
            3. Perfect Binary Tree
            4. Balanced Binary Tree
            
            Traversals:
            - Pre-order: Root → Left → Right
            - In-order: Left → Root → Right
            - Post-order: Left → Right → Root
            - Level-order (BFS)
            
            Important Trees:
            - Binary Search Tree (BST)
            - AVL Tree (Self-balancing)
            - Red-Black Tree
            - Trie (Prefix Tree)
            - Segment Tree
            - Fenwick Tree (Binary Indexed Tree)
            
            Common Algorithms:
            - Tree Height/Depth
            - Check if BST
            - Lowest Common Ancestor
            - Tree Diameter
            - Serialize/Deserialize
            """,
            
            "Graphs": """
            Graphs consist of vertices (nodes) connected by edges.
            
            Types:
            1. Directed vs Undirected
            2. Weighted vs Unweighted
            3. Cyclic vs Acyclic
            
            Representations:
            - Adjacency Matrix
            - Adjacency List
            - Edge List
            
            Traversal Algorithms:
            - Breadth-First Search (BFS)
            - Depth-First Search (DFS)
            
            Shortest Path Algorithms:
            - Dijkstra's (Non-negative weights)
            - Bellman-Ford (Negative weights)
            - Floyd-Warshall (All pairs)
            
            Minimum Spanning Tree:
            - Prim's Algorithm
            - Kruskal's Algorithm
            
            Topological Sort (for DAGs)
            Cycle Detection
            Strongly Connected Components (Kosaraju/Tarjan)
            """,
            
            "Dynamic Programming": """
            DP solves complex problems by breaking them into overlapping subproblems.
            
            Key Principles:
            1. Optimal Substructure
            2. Overlapping Subproblems
            
            Approaches:
            - Top-down (Memoization)
            - Bottom-up (Tabulation)
            
            Common DP Patterns:
            1. 0/1 Knapsack
            2. Unbounded Knapsack
            3. Fibonacci Pattern
            4. Longest Common Subsequence
            5. Longest Increasing Subsequence
            6. Edit Distance
            7. Palindromic Subsequences
            8. Subset Sum
            
            State Transition Steps:
            1. Define dp array meaning
            2. Find recurrence relation
            3. Initialize base cases
            4. Determine traversal order
            5. Return result
            """
        }
        
        documents = []
        for topic, content in dsa_topics.items():
            documents.append(Document(
                page_content=content,
                metadata={"topic": topic, "type": "concept"}
            ))
        
        # Add problems
        problems = self._get_sample_problems()
        for problem in problems:
            doc_content = f"""
            Problem: {problem.title}
            Difficulty: {problem.difficulty.value}
            Topic: {problem.topic.value}
            
            Description: {problem.description}
            
            Optimal Solution Approach: {problem.explanation}
            
            Time Complexity: {problem.time_complexity}
            Space Complexity: {problem.space_complexity}
            
            Similar Problems: {', '.join(problem.similar_problems)}
            """
            documents.append(Document(
                page_content=doc_content,
                metadata={
                    "type": "problem",
                    "topic": problem.topic.value,
                    "difficulty": problem.difficulty.value,
                    "problem_id": problem.id
                }
            ))
        
        return documents
    
    def _get_sample_problems(self) -> List[DSAProblem]:
        """Generate sample DSA problems"""
        return [
            DSAProblem(
                id="two_sum",
                title="Two Sum",
                description="Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target.",
                topic=Topic.ARRAYS,
                difficulty=Difficulty.EASY,
                code_template="def twoSum(nums: List[int], target: int) -> List[int]:",
                test_cases=[
                    {"input": {"nums": [2,7,11,15], "target": 9}, "output": [0,1]},
                    {"input": {"nums": [3,2,4], "target": 6}, "output": [1,2]},
                ],
                optimal_solution="Use hash map to store number-index pairs, check complement",
                time_complexity="O(n)",
                space_complexity="O(n)",
                hints=[
                    "Think about what information you need to find quickly",
                    "Can you use extra space to speed up lookups?",
                    "What's the complement of each number?"
                ],
                explanation="Use a hash map to store each number and its index. For each number, calculate complement = target - num. If complement exists in hash map, return indices.",
                similar_problems=["Three Sum", "Four Sum", "Two Sum II - Sorted Array"]
            ),
            DSAProblem(
                id="reverse_linked_list",
                title="Reverse Linked List",
                description="Given the head of a singly linked list, reverse the list and return the new head.",
                topic=Topic.LINKED_LISTS,
                difficulty=Difficulty.EASY,
                code_template="def reverseList(head: Optional[ListNode]) -> Optional[ListNode]:",
                test_cases=[
                    {"input": {"head": [1,2,3,4,5]}, "output": [5,4,3,2,1]},
                ],
                optimal_solution="Iterative reversal with three pointers",
                time_complexity="O(n)",
                space_complexity="O(1)",
                hints=[
                    "Think about what happens to the next pointer of each node",
                    "You need to keep track of previous node",
                    "Draw it out step by step"
                ],
                explanation="Use three pointers: prev, curr, next. At each step, save curr.next, point curr to prev, move prev to curr, move curr to saved next.",
                similar_problems=["Reverse Linked List II", "Palindrome Linked List"]
            ),
            DSAProblem(
                id="binary_tree_inorder",
                title="Binary Tree Inorder Traversal",
                description="Given the root of a binary tree, return the inorder traversal of its nodes' values.",
                topic=Topic.TREES,
                difficulty=Difficulty.EASY,
                code_template="def inorderTraversal(root: Optional[TreeNode]) -> List[int]:",
                test_cases=[
                    {"input": {"root": [1,None,2,3]}, "output": [1,3,2]},
                ],
                optimal_solution="Recursive or iterative using stack",
                time_complexity="O(n)",
                space_complexity="O(n)",
                hints=[
                    "Inorder traversal visits left, then root, then right",
                    "Can you do it recursively?",
                    "For iterative approach, think about using a stack"
                ],
                explanation="Recursive: Visit left subtree, add root value, visit right subtree. Iterative: Use stack to simulate recursion, push left nodes first.",
                similar_problems=["Preorder Traversal", "Postorder Traversal", "Level Order Traversal"]
            ),
            DSAProblem(
                id="course_schedule",
                title="Course Schedule",
                description="There are a total of numCourses courses labeled from 0 to numCourses-1. Given prerequisites, determine if you can finish all courses.",
                topic=Topic.GRAPHS,
                difficulty=Difficulty.MEDIUM,
                code_template="def canFinish(numCourses: int, prerequisites: List[List[int]]) -> bool:",
                test_cases=[
                    {"input": {"numCourses": 2, "prerequisites": [[1,0]]}, "output": True},
                    {"input": {"numCourses": 2, "prerequisites": [[1,0],[0,1]]}, "output": False},
                ],
                optimal_solution="Topological sort using Kahn's algorithm or DFS cycle detection",
                time_complexity="O(V + E)",
                space_complexity="O(V + E)",
                hints=[
                    "This is a graph problem with courses as nodes and prerequisites as edges",
                    "Think about detecting cycles in a directed graph",
                    "Topological sorting might help"
                ],
                explanation="Model as directed graph. If cycle exists, cannot finish. Use Kahn's algorithm (BFS) with indegree calculation or DFS with three colors.",
                similar_problems=["Course Schedule II", "Alien Dictionary"]
            ),
            DSAProblem(
                id="coin_change",
                title="Coin Change",
                description="Given coins of different denominations and a total amount, find the fewest number of coins needed.",
                topic=Topic.DYNAMIC_PROGRAMMING,
                difficulty=Difficulty.MEDIUM,
                code_template="def coinChange(coins: List[int], amount: int) -> int:",
                test_cases=[
                    {"input": {"coins": [1,2,5], "amount": 11}, "output": 3},
                    {"input": {"coins": [2], "amount": 3}, "output": -1},
                ],
                optimal_solution="Dynamic programming bottom-up approach",
                time_complexity="O(amount * n)",
                space_complexity="O(amount)",
                hints=[
                    "This is an unbounded knapsack problem",
                    "Think about subproblems: min coins for smaller amounts",
                    "Initialize dp[0] = 0"
                ],
                explanation="dp[i] = min coins for amount i. For each coin, dp[i] = min(dp[i], dp[i-coin] + 1). Return dp[amount] if not INF.",
                similar_problems=["Coin Change II", "Minimum Cost For Tickets"]
            )
        ]
    
    def _create_vector_store(self):
        """Create vector store for RAG"""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        
        splits = text_splitter.split_documents(self.documents)
        vector_store = FAISS.from_documents(splits, self.embeddings)
        return vector_store
    
    def query(self, question: str, topic: str = None) -> List[Document]:
        """Query the knowledge base"""
        if topic:
            # Add topic filter to query
            question = f"{question} [Topic: {topic}]"
        
        docs = self.vector_store.similarity_search(question, k=5)
        return docs