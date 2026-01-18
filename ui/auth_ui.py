import gradio as gr
import requests
import json
from typing import Dict, Optional

class AuthUI:
    def __init__(self, api_url="http://localhost:8000"):
        self.api_url = api_url
        self.current_user = None
        self.token = None
        
    def create_login_interface(self) -> gr.Blocks:
        """Create login/signup interface"""
        with gr.Blocks(title="DSA Tutor - Login", theme=gr.themes.Soft()) as login_ui:
            gr.Markdown("# DSA Tutor - Authentication")
            
            with gr.Tabs():
                with gr.TabItem("Login"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            login_username = gr.Textbox(label="Username", placeholder="Enter your username")
                            login_password = gr.Textbox(label="Password", type="password", placeholder="Enter your password")
                            login_btn = gr.Button("Login", variant="primary")
                            login_status = gr.Textbox(label="Status", interactive=False)
                    
                    login_btn.click(
                        self.handle_login,
                        inputs=[login_username, login_password],
                        outputs=[login_status]
                    )
                
                with gr.TabItem("Sign Up"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            signup_username = gr.Textbox(label="Username", placeholder="Choose a username")
                            signup_email = gr.Textbox(label="Email (optional)", placeholder="your.email@example.com")
                            signup_password = gr.Textbox(label="Password", type="password", placeholder="Create a password")
                            signup_confirm = gr.Textbox(label="Confirm Password", type="password", placeholder="Confirm password")
                            signup_fullname = gr.Textbox(label="Full Name (optional)", placeholder="Your full name")
                            
                            experience_level = gr.Dropdown(
                                choices=["Beginner", "Intermediate", "Advanced"],
                                value="Beginner",
                                label="Experience Level"
                            )
                            
                            signup_btn = gr.Button("Create Account", variant="primary")
                            signup_status = gr.Textbox(label="Status", interactive=False)
                    
                    signup_btn.click(
                        self.handle_signup,
                        inputs=[signup_username, signup_email, signup_password, 
                                signup_confirm, signup_fullname, experience_level],
                        outputs=[signup_status]
                    )
            
            return login_ui
    
    def handle_login(self, username: str, password: str) -> str:
        """Handle login request"""
        try:
            response = requests.post(
                f"{self.api_url}/api/login",
                json={"username": username, "password": password}
            )
            
            if response.status_code == 200:
                data = response.json()
                self.current_user = {"username": username, "user_id": data["user_id"]}
                self.token = data["token"]
                return "Login successful! Redirecting..."
            else:
                return f"Login failed: {response.json().get('detail', 'Unknown error')}"
                
        except Exception as e:
            return f"Connection error: {str(e)}"
    
    def handle_signup(self, username: str, email: str, password: str, 
                     confirm_password: str, full_name: str, experience_level: str) -> str:
        """Handle signup request"""
        if password != confirm_password:
            return "Passwords do not match!"
        
        if len(password) < 6:
            return "Password must be at least 6 characters!"
        
        try:
            response = requests.post(
                f"{self.api_url}/api/signup",
                json={
                    "username": username,
                    "email": email if email else None,
                    "password": password,
                    "full_name": full_name if full_name else None,
                    "experience_level": experience_level.lower()
                }
            )
            
            if response.status_code == 200:
                data = response.json()
                self.current_user = {"username": username, "user_id": data["user_id"]}
                self.token = data["token"]
                return "Account created successfully! Redirecting..."
            else:
                return f"Signup failed: {response.json().get('detail', 'Unknown error')}"
                
        except Exception as e:
            return f"Connection error: {str(e)}"
    
    def get_auth_header(self) -> Dict:
        """Get authentication header for API requests"""
        if self.token:
            return {"Authorization": f"Bearer {self.token}"}
        return {}