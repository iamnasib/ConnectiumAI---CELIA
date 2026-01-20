import os
import httpx
import re 
import unicodedata
from PIL import Image

from fastapi import FastAPI,BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

from dotenv import load_dotenv
import logging

from pipecat.transports.daily.transport import DailyTransport, DailyParams
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.services.openai.realtime.llm  import OpenAIRealtimeLLMService
from pipecat.services.openai.realtime.events import (
    SessionProperties,
    AudioConfiguration,
    AudioInput,
    InputAudioTranscription,
    InputAudioNoiseReduction,
    AudioOutput,
    SemanticTurnDetection,
    ConversationItemCreateEvent,
    ConversationItem,
    ItemContent,
    ResponseCreateEvent)
from pipecat.processors.user_idle_processor import UserIdleProcessor
from pipecat.observers.loggers.transcription_log_observer import TranscriptionLogObserver
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair,LLMUserAggregatorParams
from pipecat.frames.frames import (
    LLMRunFrame,
    EndFrame,
    Frame,
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    OutputImageRawFrame,
    SpriteFrame
    )
from pipecat.processors.frame_processor import FrameProcessor,FrameDirection
# from pipecat.turns.mute import MuteUntilFirstBotCompleteUserMuteStrategy

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="ConnectiumAI API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your Bubble.io domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class InterviewRequest(BaseModel):
    daily_room_url:str
    daily_room_token:Optional[str]=None
    bot_prompt:str
    bot_name:str
    stop_phrase:Optional[str]=None
    bubble_webhook_callback_url:Optional[str]=None

class InterviewResponse(BaseModel):
    status: str
    message: str

class InterviewResult(BaseModel):
    room_name:str
    transcript_text: str
    stop_reason:str

sprites=[]

script_dir = os.path.dirname(__file__)
if not sprites:
    try:
        for i in range(1,12):
            full_path = os.path.join(script_dir, f"assets/ConnectiumAI_visualizer_{i}.png")
           
            with Image.open(full_path) as img:
                
                # Ensure RGB format
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                img_copy = img.copy()
                raw_img = OutputImageRawFrame(
                    image=img_copy.tobytes(),
                    size=img_copy.size,
                    format='RGB'
                )
                sprites.append(raw_img)

        

        logger.info(f"Loaded {len(sprites)} sprite frames")
    except Exception as e:
        logger.error(f"Error loading sprite frames: {e}")

    flipped=sprites[::-1]
    sprites.extend(flipped)
    quiet_frame=sprites[0]
    talking_frame=SpriteFrame(images=sprites)

class TalkingAnimation(FrameProcessor):
    def __init__(self):
        super().__init__()
        self.is_talking=False
    
    async def process_frame(self, frame:Frame, direction:FrameDirection):
        await super().process_frame(frame, direction)

        # Switch to talking animation when bot starts speaking
        if isinstance(frame,BotStartedSpeakingFrame):
            if not self.is_talking:
                await self.push_frame(talking_frame)
                self.is_talking=True
        elif isinstance(frame,BotStoppedSpeakingFrame):
            await self.push_frame(quiet_frame)
            self.is_talking=False
        
        await self.push_frame(frame,direction)

class InterviewBot:
    def __init__(self,
                daily_room_url,
                daily_room_token,
                bot_prompt,
                bot_name,
                stop_phrase,
                bubble_webhook_callback_url,
                 ):
        self.transcription_text=""
        self.daily_room_url = (daily_room_url or "").strip().replace("\u200b", "").replace("\ufeff", "")
        self.daily_room_token=daily_room_token
        self.bot_prompt=bot_prompt
        self.bot_name=bot_name
        self.stop_phrase=stop_phrase
        self.bubble_webhook_callback_url=bubble_webhook_callback_url
        self.reason="Finished"
    
    def normalize_text(self,text: str) -> str:
        text = unicodedata.normalize("NFKD", text)
        text = text.lower()
        text = re.sub(r"[^\w\s]", "", text)  # remove punctuation
        text = re.sub(r"\s+", " ", text).strip()
        return text
    

    async def run(self):
        try:

            transport=DailyTransport(
                self.daily_room_url,
                self.daily_room_token,
                self.bot_name,
                DailyParams(
                    audio_out_enabled=True,
                    audio_in_enabled=True,
                    video_out_enabled=True,
                    video_out_width=1024,
                    video_out_height=576,
                    vad_analyzer=SileroVADAnalyzer()
                )
            )
   
            session_properties = SessionProperties(
                instructions=self.bot_prompt,
                voice="ash",
                audio=AudioConfiguration(
                    input=AudioInput(
                        transcription=InputAudioTranscription(model="whisper-1"),
                        turn_detection=SemanticTurnDetection(type="semantic_vad",eagerness="medium"),
                        noise_reduction=InputAudioNoiseReduction(type='near_field')
                    ), output=AudioOutput(
                        voice="marin"
                    )
                )
            )

            llm=OpenAIRealtimeLLMService(
                api_key=os.getenv('OPENAI_API_KEY'),
                session_properties=session_properties)

            context=LLMContext()
            user_aggregator,assistant_aggregator=LLMContextAggregatorPair(
                context,
                # user_params=LLMUserAggregatorParams(
                #     user_mute_strategies=[
                #         MuteUntilFirstBotCompleteUserMuteStrategy(),
                #     ],
                # ),
            )

            
            def make_idle_handler(llm_service):
                async def handle_user_idle(user_idle: UserIdleProcessor, retry_count: int) -> bool:
                    if retry_count < 3:
                        message = "The user has been quiet. Politely and briefly ask if they're still there."
                    else:
                        message = "The user is still completely inactive. End the conversation politely."
                    
                    logger.info(f"User idle (attempt {retry_count})")
                    
                    # Step 1: Create the conversation item
                    conversation_item = ConversationItem(
                        type="message",
                        role="user",
                        content=[
                            ItemContent(
                                type="input_text",
                                text=message
                            )
                        ]
                    )
                    
                    # Step 2: Create the event with the item
                    create_event = ConversationItemCreateEvent(
                        item=conversation_item
                    )
                    
                    # Step 3: Send the conversation item
                    await llm_service.send_client_event(create_event)
                    
                    # Step 4: Trigger the model to respond
                    response_event = ResponseCreateEvent()
                    await llm_service.send_client_event(response_event)
                    
                    if retry_count < 3: 
                        return True  # Continue waiting
                    else:
                        self.reason="User Inactive"
                        logger.info("User inactive for too long. Ending conversation.")
                        task.queue_frames([EndFrame()])
                        return False
                
                return handle_user_idle
            
            user_idle = UserIdleProcessor(
                callback=make_idle_handler(llm), 
                timeout=10.0
            )

            talking_animation=TalkingAnimation()
            pipeline=Pipeline(
                [
                    transport.input(),
                    user_aggregator,
                    user_idle,
                    llm,
                    talking_animation,
                    transport.output(),
                    assistant_aggregator
                ]
            )
            
            task = PipelineTask(
                    pipeline,
                    params=PipelineParams(
                        enable_metrics=True,
                        enable_usage_metrics=True,
                    ),
                    observers=[TranscriptionLogObserver()]
                )

            @transport.event_handler("on_first_participant_joined")
            async def on_first_participant_joined(transport, participant):
                participant_name = participant.get("info", {}).get("userName", "Unknown")
                logger.info(f"First Participant joined: {participant_name} (ID: {participant['id']})")
                try:
                    await task.queue_frames([LLMRunFrame()])
                    logger.info("LLMRunFrame queued after participant joined")
                except Exception as e:
                    logger.error(f"Failed to queue LLMRunFrame on participant join: {e}")

            @user_aggregator.event_handler("on_user_turn_stopped")
            async def on_user_turn_stopped(aggregator, strategy, message):
                if message.content:
                    line = f"[{message.timestamp}] - User: {message.content}"
                    self.transcription_text+=f"{line}\n"
                # print(f"User turn stopped: {message.content}")

            @assistant_aggregator.event_handler("on_assistant_turn_stopped")
            async def on_assistant_turn_stopped(aggregator, message):
                if message.content:
                    line=f"[{message.timestamp}] - AI Assistant: {message.content}"
                    self.transcription_text+=f"{line}\n"

                    if self.stop_phrase and (self.normalize_text(self.stop_phrase) in self.normalize_text(message.content)):
                        logger.info(f"Stop Phrase ({self.stop_phrase}) spoken by {self.bot_name}. Ending Now...")
                        #await task.queue_frames([EndFrame()])
                        await task.stop_when_done()

                           
            @transport.event_handler("on_client_disconnected")
            async def on_client_disconnected(transport, client):
                logger.info("Client disconnected")
                await task.cancel()

            runner = PipelineRunner(handle_sigint=False)

            await runner.run(task)

            if self.bubble_webhook_callback_url and self.reason != "User Inactive":
                await self.send_results_back(self.reason)

        except Exception as e:
            logger.error(f"Error in interview bot: {e}")
            # await self._send_error_to_bubble(str(e))
    async def send_results_back(self,reason:str="Finished"):
        try:
            logger.info(f"Sending results back for {self.bubble_webhook_callback_url}")

            result=InterviewResult(
                room_name=self.daily_room_url.split("/")[-1],
                transcript_text=self.transcription_text,
                stop_reason=reason
            )

            async with httpx.AsyncClient(timeout=30) as client:
                response= await client.post(
                    self.bubble_webhook_callback_url,
                    json=result.model_dump(mode="json"),
                    headers={"Content-Type": "application/json"}
                )

                if response.status_code ==200:
                    logger.info(f"Successfully sent result back to: {self.bubble_webhook_callback_url}")
                else:
                    logger.error(f"Failed to send results: {response.status_code} - {response.text}")
        except Exception as e:
            logger.error(f"Error in sending results: {e}")
            # await self._send_error_to_bubble(str(e))



@app.get("/")
def main():
    return {"status": "ok"}

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post('/start-interview', response_model=InterviewResponse)
async def start_interview(request:InterviewRequest,background_tasks:BackgroundTasks):
    try:
        bot = InterviewBot(
            daily_room_url=request.daily_room_url,
            daily_room_token=request.daily_room_token,
            bot_prompt=request.bot_prompt,
            bot_name=request.bot_name,
            stop_phrase=request.stop_phrase,
            bubble_webhook_callback_url=request.bubble_webhook_callback_url
        )

        

        background_tasks.add_task(bot.run)
        
        return InterviewResponse(
            status="started",
            message=f"Interview bot joining meeting for {request.daily_room_url}"
        )
        
    except Exception as e:
        logger.error(f"Error starting interview: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 7860))
    logger.info(f"Starting server on port {port}")  # Add this line
    logger.info(f"PORT environment variable: {os.getenv('PORT')}")  # Add this line
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
