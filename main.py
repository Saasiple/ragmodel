from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import openai
import os
from datetime import datetime

app = FastAPI(title="Chatbot APIs", version="1.0.0")

# Set your OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Pydantic models for request/response
class Message(BaseModel):
    role: str  # "user" or "assistant"
    content: str
    timestamp: Optional[datetime] = None

class ChatSummaryRequest(BaseModel):
    messages: List[Message]
    max_length: Optional[int] = 150

class ChatSummaryResponse(BaseModel):
    summary: str
    message_count: int
    conversation_duration: Optional[str] = None

class ReplyOptionsRequest(BaseModel):
    messages: List[Message]
    context: Optional[str] = None
    tone: Optional[str] = "helpful"  # helpful, casual, professional, friendly

class ReplyOption(BaseModel):
    option_number: int
    reply: str
    tone_description: str

class ReplyOptionsResponse(BaseModel):
    options: List[ReplyOption]
    original_message: str

@app.post("/analyze-chat", response_model=ChatSummaryResponse)
async def analyze_chat_summary(request: ChatSummaryRequest):
    """
    Analyze chat messages and provide a summary
    """
    try:
        if not request.messages:
            raise HTTPException(status_code=400, detail="No messages provided")
        
        # Prepare the conversation for summarization
        conversation_text = ""
        for msg in request.messages:
            conversation_text += f"{msg.role.capitalize()}: {msg.content}\n"
        
        # Create the prompt for summarization
        summary_prompt = f"""
        Please provide a concise summary of the following conversation in {request.max_length} characters or less.
        Focus on the main topics discussed, key decisions made, and important information exchanged.
        
        Conversation:
        {conversation_text}
        
        Summary:
        """
        
        # Call OpenAI API for summarization
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that creates concise, accurate summaries of conversations."},
                {"role": "user", "content": summary_prompt}
            ],
            max_tokens=100,
            temperature=0.3
        )
        
        summary = response.choices[0].message.content.strip()
        
        # Calculate conversation duration if timestamps are available
        duration = None
        timestamped_messages = [msg for msg in request.messages if msg.timestamp]
        if len(timestamped_messages) >= 2:
            start_time = min(msg.timestamp for msg in timestamped_messages)
            end_time = max(msg.timestamp for msg in timestamped_messages)
            duration = str(end_time - start_time)
        
        return ChatSummaryResponse(
            summary=summary,
            message_count=len(request.messages),
            conversation_duration=duration
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating summary: {str(e)}")

@app.post("/generate-replies", response_model=ReplyOptionsResponse)
async def generate_reply_options(request: ReplyOptionsRequest):
    """
    Generate 3 AI reply options based on the chat conversation
    """
    try:
        if not request.messages:
            raise HTTPException(status_code=400, detail="No messages provided")
        
        # Get the last message (what we're replying to)
        last_message = request.messages[-1]
        
        # Prepare conversation context
        conversation_context = ""
        for msg in request.messages[-5:]:  # Use last 5 messages for context
            conversation_context += f"{msg.role.capitalize()}: {msg.content}\n"
        
        # Create the prompt for generating reply options
        reply_prompt = f"""
        Based on the following conversation, generate 3 different reply options with varying tones and styles.
        The replies should be appropriate responses to the last message.
        
        Context: {request.context or "General conversation"}
        Desired tone: {request.tone}
        
        Conversation:
        {conversation_context}
        
        Please provide 3 different reply options:
        1. A {request.tone} and direct response
        2. A more detailed and explanatory response  
        3. A warm and engaging response
        
        Format each reply clearly and keep them concise but helpful.
        """
        
        # Call OpenAI API for reply generation
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that generates appropriate reply options for conversations. Provide exactly 3 distinct reply options with different styles."},
                {"role": "user", "content": reply_prompt}
            ],
            max_tokens=300,
            temperature=0.7
        )
        
        generated_text = response.choices[0].message.content.strip()
        
        # Parse the generated replies (this is a simple parsing - you might want to improve this)
        reply_lines = [line.strip() for line in generated_text.split('\n') if line.strip()]
        
        # Extract the 3 options
        options = []
        option_descriptions = ["Direct & Helpful", "Detailed & Explanatory", "Warm & Engaging"]
        
        current_option = 1
        current_reply = ""
        
        for line in reply_lines:
            if any(str(i) in line for i in [1, 2, 3]) and (line.startswith(('1.', '2.', '3.')) or line.startswith(('Option', 'Reply'))):
                if current_reply and current_option <= 3:
                    options.append(ReplyOption(
                        option_number=current_option,
                        reply=current_reply.strip(),
                        tone_description=option_descriptions[current_option-1] if current_option <= 3 else "Alternative"
                    ))
                    current_option += 1
                    current_reply = ""
                current_reply += line.split('.', 1)[-1].strip() if '.' in line else line
            else:
                current_reply += " " + line if current_reply else line
        
        # Add the last option if it exists
        if current_reply and current_option <= 3:
            options.append(ReplyOption(
                option_number=current_option,
                reply=current_reply.strip(),
                tone_description=option_descriptions[current_option-1] if current_option <= 3 else "Alternative"
            ))
        
        # Fallback: if parsing failed, create simple options
        if len(options) < 3:
            fallback_replies = generated_text.split('\n\n') if '\n\n' in generated_text else [generated_text]
            options = []
            for i, reply in enumerate(fallback_replies[:3], 1):
                if reply.strip():
                    options.append(ReplyOption(
                        option_number=i,
                        reply=reply.strip(),
                        tone_description=option_descriptions[i-1] if i <= 3 else f"Option {i}"
                    ))
        
        # Ensure we have exactly 3 options
        while len(options) < 3:
            options.append(ReplyOption(
                option_number=len(options) + 1,
                reply=f"Thank you for your message. I'd be happy to help you with that.",
                tone_description="Default Response"
            ))
        
        return ReplyOptionsResponse(
            options=options[:3],  # Ensure only 3 options
            original_message=last_message.content
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating replies: {str(e)}")

@app.get("/")
async def root():
    return {
        "message": "Chatbot API is running!",
        "endpoints": {
            "analyze_chat": "/analyze-chat",
            "generate_replies": "/generate-replies"
        }
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)