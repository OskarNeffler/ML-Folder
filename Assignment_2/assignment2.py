import os
import json
from typing import List, Dict, Any, Optional
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables (create a .env file with your OpenAI API key)
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Cost saving settings - defaults that can be overridden in .env file
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "gpt-3.5-turbo")  # Use cheaper model by default
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "400"))  # Limit token usage
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))
USE_CHEAPER_MODEL = os.getenv("USE_CHEAPER_MODEL", "true").lower() == "true"  # Set to False if you want to use GPT-4 for all responses

# Print the current settings
print("=== Cost Saving Settings ===")
print(f"Default Model: {DEFAULT_MODEL}")
print(f"Max Tokens: {MAX_TOKENS}")
print(f"Use Cheaper Model: {USE_CHEAPER_MODEL}")
print("=========================\n")

class AIAgent:
    def __init__(self, name: str, persona: str, system_prompt: str, use_gpt4: bool = False):
        self.name = name
        self.persona = persona
        self.conversation_history = [{"role": "system", "content": system_prompt}]
        self.use_gpt4 = use_gpt4
        
    def add_message(self, role: str, content: str):
        """Add a message to the agent's conversation history."""
        self.conversation_history.append({"role": role, "content": content})
        
    def generate_response(self, prompt: Optional[str] = None) -> str:
        """Generate a response from the agent based on conversation history."""
        if prompt:
            self.add_message("user", prompt)
            
        try:
            # Select model based on settings
            if USE_CHEAPER_MODEL and not self.use_gpt4:
                model = DEFAULT_MODEL  # Use cheaper model
            else:
                model = "gpt-4"  # Use GPT-4 when specifically needed
                
            response = client.chat.completions.create(
                model=model,
                messages=self.conversation_history,
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE
            )
            message_content = response.choices[0].message.content
            self.add_message("assistant", message_content)
            return message_content
        except Exception as e:
            print(f"Error generating response for {self.name}: {e}")
            return f"[Error generating response: {e}]"
    
    def get_last_response(self) -> str:
        """Get the most recent response from the agent."""
        for message in reversed(self.conversation_history):
            if message["role"] == "assistant":
                return message["content"]
        return ""
    
    def __str__(self) -> str:
        return f"{self.name} ({self.persona})"


class DebateManager:
    def __init__(self, agents: List[AIAgent], moderator: AIAgent, budget_mode: bool = False):
        self.agents = agents
        self.moderator = moderator
        self.full_transcript = []
        self.budget_mode = budget_mode
        
    def add_to_transcript(self, speaker: str, message: str):
        """Add a message to the full debate transcript."""
        self.full_transcript.append({"speaker": speaker, "message": message})
        
    def format_transcript_for_agent(self, exclude_agent: Optional[AIAgent] = None) -> str:
        """Format the transcript in a way that can be sent to an agent."""
        formatted = "Current debate transcript:\n\n"
        
        # In budget mode, only include the last few exchanges to save tokens
        entries = self.full_transcript
        if self.budget_mode:
            # Only include the last 6 exchanges to save on tokens
            entries = self.full_transcript[-12:] if len(self.full_transcript) > 12 else self.full_transcript
            
        for entry in entries:
            if exclude_agent and entry["speaker"] == exclude_agent.name:
                continue
            formatted += f"{entry['speaker']}: {entry['message']}\n\n"
        return formatted
    
    def conduct_initial_statements(self):
        """Have each agent give their initial platform advocacy."""
        print("=== INITIAL PLATFORM STATEMENTS ===\n")
        
        # Moderator introduces the debate
        intro_prompt = "Please introduce the debate on the optimal computing platform for AI/ML coursework. Explain that each expert will give their initial statement."
        intro = self.moderator.generate_response(intro_prompt)
        print(f"Moderator: {intro}\n")
        self.add_to_transcript("Moderator", intro)
        
        # Each agent gives their initial statement
        for agent in self.agents:
            prompt = (f"Please provide your initial statement advocating for {agent.persona} as the optimal "
                     f"computing platform for a student taking an ML+AI course. Highlight your platform's "
                     f"key advantages in 2-3 paragraphs.")
            
            response = agent.generate_response(prompt)
            print(f"{agent.name}: {response}\n")
            self.add_to_transcript(agent.name, response)
    
    def conduct_technical_comparison(self, topic: str):
        """Have agents discuss a specific technical topic."""
        print(f"=== TECHNICAL COMPARISON: {topic.upper()} ===\n")
        
        # Moderator introduces the topic
        topic_intro_prompt = f"Please introduce the next topic for debate: {topic}. Ask the experts to provide their platform's strengths and weaknesses on this specific aspect."
        topic_intro = self.moderator.generate_response(topic_intro_prompt)
        print(f"Moderator: {topic_intro}\n")
        self.add_to_transcript("Moderator", topic_intro)
        
        # Each agent discusses the topic
        for agent in self.agents:
            transcript = self.format_transcript_for_agent()
            prompt = (f"Based on the debate so far:\n\n{transcript}\n\n"
                     f"Please discuss how {agent.persona} handles {topic}. "
                     f"Be specific about both strengths and limitations of your platform in this area. "
                     f"Support your claims with technical details relevant to ML/AI coursework.")
            
            response = agent.generate_response(prompt)
            print(f"{agent.name}: {response}\n")
            self.add_to_transcript(agent.name, response)
    
    def conduct_rebuttals(self):
        """Have agents rebut each other's claims."""
        print("=== REBUTTALS AND COUNTER-ARGUMENTS ===\n")
        
        # Moderator introduces the rebuttal phase
        rebuttal_prompt = "Please introduce the rebuttal phase. Ask each expert to address claims made by the others and defend their platform against criticisms."
        rebuttal_intro = self.moderator.generate_response(rebuttal_prompt)
        print(f"Moderator: {rebuttal_intro}\n")
        self.add_to_transcript("Moderator", rebuttal_intro)
        
        # Each agent offers rebuttals
        for agent in self.agents:
            transcript = self.format_transcript_for_agent()
            prompt = (f"Based on the debate so far:\n\n{transcript}\n\n"
                     f"Please address claims made by other experts that you disagree with or find misleading. "
                     f"Defend your platform ({agent.persona}) against any criticisms while being factual and respectful. "
                     f"Focus on technical inaccuracies or exaggerations in their statements.")
            
            response = agent.generate_response(prompt)
            print(f"{agent.name}: {response}\n")
            self.add_to_transcript(agent.name, response)
    
    def conduct_final_recommendations(self):
        """Have agents provide their final recommendations."""
        print("=== FINAL RECOMMENDATIONS ===\n")
        
        # Moderator introduces the final recommendations phase
        final_prompt = "Please introduce the final recommendations phase. Ask each expert to provide their balanced conclusion about which setup would be best for an ML/AI student, considering all factors discussed."
        final_intro = self.moderator.generate_response(final_prompt)
        print(f"Moderator: {final_intro}\n")
        self.add_to_transcript("Moderator", final_intro)
        
        # Each agent gives final recommendation
        for agent in self.agents:
            transcript = self.format_transcript_for_agent()
            prompt = (f"Based on the entire debate:\n\n{transcript}\n\n"
                     f"Please provide your final recommendation for an ML/AI student. "
                     f"Be balanced and honest, acknowledging when other platforms might be better for certain tasks. "
                     f"Consider student budget constraints, learning curve, and the full range of ML/AI tasks mentioned. "
                     f"You can suggest hybrid approaches if appropriate.")
            
            response = agent.generate_response(prompt)
            print(f"{agent.name}: {response}\n")
            self.add_to_transcript(agent.name, response)
        
        # Moderator summarizes the debate
        summary_prompt = f"Based on the entire debate transcript:\n\n{self.format_transcript_for_agent()}\n\nPlease summarize the key points from all experts and provide a balanced conclusion about the optimal computing platforms for ML/AI students considering different scenarios and needs."
        summary = self.moderator.generate_response(summary_prompt)
        print(f"Moderator: {summary}\n")
        self.add_to_transcript("Moderator", summary)
    
    def save_transcript(self, filename: str = "debate_transcript.json"):
        """Save the full transcript to a file."""
        with open(filename, 'w') as f:
            json.dump(self.full_transcript, f, indent=2)
        print(f"Transcript saved to {filename}")
    
    def conduct_full_debate(self):
        """Run the complete debate from start to finish."""
        # Initial statements
        self.conduct_initial_statements()
        
        # Technical comparisons on specific topics
        topics = [
            "GPU acceleration and compute requirements", 
            "Development environment setup and maintenance",
            "Software compatibility for ML frameworks",
            "Cost considerations (hardware and cloud)",
            "Performance benchmarks for ML workloads",
            "Local vs. cloud trade-offs"
        ]
        
        for topic in topics:
            self.conduct_technical_comparison(topic)
        
        # Rebuttals
        self.conduct_rebuttals()
        
        # Final recommendations
        self.conduct_final_recommendations()
        
        # Save the transcript
        self.save_transcript()
    
    def conduct_budget_debate(self):
        """Run a shorter debate with fewer topics to save API costs."""
        print("Running budget-friendly debate with fewer topics...\n")
        
        # Initial statements
        self.conduct_initial_statements()
        
        # Technical comparisons on just 2 important topics
        budget_topics = [
            "Development environment setup and maintenance",
            "Cost considerations (hardware and cloud)"
        ]
        
        for topic in budget_topics:
            self.conduct_technical_comparison(topic)
        
        # Skip rebuttals to save costs
        
        # Final recommendations
        self.conduct_final_recommendations()
        
        # Save the transcript
        self.save_transcript("budget_debate_transcript.json")


def create_agents() -> List[AIAgent]:
    """Create and return the list of AI agents with their personas."""
    
    # System prompts for each agent
    linux_prompt = """You are a Linux Specialist with extensive experience in ML/AI development environments.
You advocate for Linux as the optimal platform for ML/AI coursework, highlighting its strengths in performance, 
customization, and open-source ecosystem. You have deep knowledge of GPU configuration, package management,
and kernel optimizations for ML workloads. While you prefer Linux, you should be factual and acknowledge
its limitations where appropriate. Focus on technical accuracy while maintaining your stance as a Linux advocate."""

    macos_prompt = """You are a macOS Advocate with extensive experience in ML/AI development workflows.
You advocate for macOS as the optimal platform for ML/AI coursework, highlighting its strengths in user experience,
hardware-software integration, and developer tools. You have deep knowledge of Mac-specific ML optimizations,
compatibility with popular frameworks, and the advantages of Unix-based systems with consumer-friendly interfaces.
While you prefer macOS, you should be factual and acknowledge its limitations where appropriate.
Focus on technical accuracy while maintaining your stance as a macOS advocate."""

    windows_prompt = """You are a Windows Expert with extensive experience in ML/AI development environments.
You advocate for Windows as the optimal platform for ML/AI coursework, highlighting its strengths in software
compatibility, enterprise support, and recent improvements for developers. You have deep knowledge of WSL,
DirectML, CUDA on Windows, and integration with cloud services. While you prefer Windows, you should be
factual and acknowledge its limitations where appropriate. Focus on technical accuracy while maintaining
your stance as a Windows advocate."""

    cloud_prompt = """You are a Cloud Solutions Architect with extensive experience in ML/AI cloud deployments.
You advocate for cloud-based solutions as the optimal platform for ML/AI coursework, highlighting strengths
in scalability, cost-effectiveness for students, and access to specialized hardware. You have deep knowledge of
various cloud providers' ML services, Jupyter notebook integrations, and serverless ML workflows. While you
prefer cloud solutions, you should be factual and acknowledge limitations where appropriate. Focus on
technical accuracy while maintaining your stance as a cloud solutions advocate."""

    ml_practitioner_prompt = """You are an ML Practitioner with extensive experience across various platforms.
You take a pragmatic approach to platform selection, focusing on what works best for specific ML/AI tasks rather
than advocating for a single platform. You have deep knowledge of the technical requirements for various ML
workloads and can evaluate platform trade-offs objectively. You focus on technical accuracy and practical
considerations for students learning ML/AI."""

    moderator_prompt = """You are a neutral Debate Moderator facilitating a discussion between experts advocating
for different computing platforms for ML/AI coursework. Your role is to guide the conversation, introduce topics,
ask clarifying questions, and ensure all participants get equal speaking time. You should remain impartial and
not advocate for any specific platform. Focus on extracting useful insights from the debate and summarizing key points."""
    
    # Create the agents
    # The moderator and ML practitioner use GPT-4 for better quality
    # The platform advocates use GPT-3.5-Turbo to save costs
    agents = [
        AIAgent("Linux Expert", "Linux", linux_prompt),
        AIAgent("macOS Expert", "macOS", macos_prompt),
        AIAgent("Windows Expert", "Windows", windows_prompt),
        AIAgent("Cloud Expert", "cloud solutions", cloud_prompt),
        AIAgent("ML Practitioner", "platform-agnostic ML solutions", ml_practitioner_prompt, use_gpt4=True)
    ]
    
    # Moderator uses GPT-4 for better quality guidance
    moderator = AIAgent("Moderator", "debate moderator", moderator_prompt, use_gpt4=True)
    
    return agents, moderator


def main():
    """Main function to run the debate."""
    print("Starting Multi-Agent Platform Debate...\n")
    
    # Ask user if they want to run in budget mode
    budget_mode = input("Do you want to run in budget-saving mode? (y/n, default=y): ").lower() != 'n'
    
    # Create agents and moderator
    agents, moderator = create_agents()
    
    # If budget mode, use a subset of agents
    if budget_mode:
        # Use only Linux, macOS, and Cloud experts for a balanced but cheaper debate
        selected_agents = [agents[0], agents[1], agents[3]]
        print(f"\nRunning in budget mode with {len(selected_agents)} agents:")
        for agent in selected_agents:
            print(f"- {agent}")
        print("")
        
        debate_manager = DebateManager(selected_agents, moderator, budget_mode=True)
        debate_manager.conduct_budget_debate()
    else:
        # Use all agents for a full debate
        print("\nRunning full debate with all agents (this will use more API credits)\n")
        debate_manager = DebateManager(agents, moderator)
        debate_manager.conduct_full_debate()


if __name__ == "__main__":
    main()
