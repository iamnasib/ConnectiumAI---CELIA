import os
import pickle
import httpx
import re 
import unicodedata
from fastapi import FastAPI,BackgroundTasks, HTTPException,Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional,Annotated
from config import IS_PRODUCTION, IS_DEVELOPMENT,_get_fly_config,_get_fly_headers
from fastapi.security import HTTPBearer,HTTPAuthorizationCredentials

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
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair,LLMUserAggregatorParams,UserTurnStoppedMessage,AssistantTurnStoppedMessage
from pipecat.frames.frames import (
    LLMRunFrame,
    EndFrame,
    Frame,
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    SpriteFrame
    )
from pipecat.processors.frame_processor import FrameProcessor,FrameDirection
from pipecat.turns.user_mute import AlwaysUserMuteStrategy

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

security= HTTPBearer()

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
    bot_function:str

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
#     try:
#         for i in range(1,12):
#             full_path = os.path.join(script_dir, f"assets/ConnectiumAI_visualizer_{i}.bin")
           
#             with open(full_path,'rb') as img:
#                 raw_bytes=img.read()
#                 raw_img = OutputImageRawFrame(
#                     image=raw_bytes,
#                     size=(1024, 576),
#                     format='RGB'
#                 )
#                 sprites.append(raw_img)

        

#         logger.info(f"Loaded {len(sprites)} sprite frames")
#     except Exception as e:
#         logger.error(f"Error loading sprite frames: {e}")

    full_path = os.path.join(script_dir, f"assets")

    with open(f"{full_path}/sprites_cache.pkl", "rb") as f:
        sprites = pickle.load(f)

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
                bot_function
                 ):
        self.transcription_text=""
        self.daily_room_url = (daily_room_url or "").strip().replace("\u200b", "").replace("\ufeff", "")
        self.daily_room_token=daily_room_token
        self.bot_prompt=bot_prompt
        self.bot_name=bot_name
        self.stop_phrase=stop_phrase
        self.bubble_webhook_callback_url=bubble_webhook_callback_url
        self.bot_function=bot_function
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
                audio=AudioConfiguration(
                    input=AudioInput(
                        transcription=InputAudioTranscription(model="whisper-1"),
                        turn_detection=SemanticTurnDetection(),
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
            if self.bot_function == "disclaimer_bot":
                user_aggregator,assistant_aggregator=LLMContextAggregatorPair(
                    context,
                    user_params=LLMUserAggregatorParams(
                        user_mute_strategies=[
                            AlwaysUserMuteStrategy(),
                        ],
                    ),
                )
            else:
                user_aggregator,assistant_aggregator=LLMContextAggregatorPair(context)

            
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
                        enable_usage_metrics=True
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
            async def on_user_turn_stopped(aggregator, strategy, message:UserTurnStoppedMessage):                
                if message.content:
                    line = f"[{message.timestamp}] - User: {message.content}"
                    self.transcription_text += f"{line}\n"
                    logger.info(f"✅ Added to transcript: {line}")
                else:
                    logger.warning("⚠️ USER TURN STOPPED but message.content is EMPTY!")
                    logger.warning(f"⚠️ Full message object: {vars(message) if hasattr(message, '__dict__') else str(message)}")

            @assistant_aggregator.event_handler("on_assistant_turn_stopped")
            async def on_assistant_turn_stopped(aggregator, message:AssistantTurnStoppedMessage):
                if message.content:
                    line=f"[{message.timestamp}] - AI Assistant: {message.content}"
                    self.transcription_text+=f"{line}\n"

                    if self.stop_phrase and (self.normalize_text(self.stop_phrase) in self.normalize_text(message.content)):
                        logger.info(f"Stop Phrase ({self.stop_phrase}) spoken by {self.bot_name}. Ending Now...")
                        #await task.queue_frames([EndFrame()])
                        await task.cancel()

            @user_aggregator.event_handler("on_user_mute_started")
            async def on_user_mute_started(aggregator):
                logger.info(f"User mute started")

            @user_aggregator.event_handler("on_user_mute_stopped")
            async def on_user_mute_stopped(aggregator):
                logger.info(f"User mute stopped")
                           
            @transport.event_handler("on_client_disconnected")
            async def on_client_disconnected(transport, client):
                logger.info("Client disconnected")
                await task.cancel()

            runner = PipelineRunner(handle_sigint=False)

            await runner.run(task)

            if self.bubble_webhook_callback_url and self.bubble_webhook_callback_url != 'null' and self.reason != "User Inactive":
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


async def run_bot_worker():
    try:
        bot = InterviewBot(
            daily_room_url=os.getenv("BOT_ROOM_URL"),
            daily_room_token=os.getenv("BOT_ROOM_TOKEN"),
            bot_prompt=os.getenv("BOT_PROMPT"),
            bot_name=os.getenv("BOT_NAME"),
            stop_phrase=os.getenv("BOT_STOP_PHRASE"),
            bubble_webhook_callback_url=os.getenv("CALLBACK_URL"),
            bot_function=os.getenv("BOT_FUNCTION")
        )

        await bot.run()
        
        
    except Exception as e:
        logger.error(f"Error starting interview: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
def main():
    return {"status": "ok"}

@app.get("/health")
def health_check():
    return {"status": "ok", "region": os.getenv("FLY_REGION")}

@app.post("/start-interview", response_model=InterviewResponse)
async def start_interview(request: InterviewRequest, token:Annotated[HTTPAuthorizationCredentials,Depends(security)], background_tasks:BackgroundTasks):
    """
        This endpoint is called by frontend . 
        It spawns a NEW Fly Machine to run the bot.
    """
    try:
        if os.getenv("BUBBLE_ACCESS_TOKEN") != token.credentials:
            raise HTTPException(status_code=403, detail="Invalid access token")
        
        if not request.daily_room_url or not request.daily_room_url.strip():
            raise HTTPException(status_code=400, detail="daily_room_url is required")
        
        if not request.bot_prompt or not request.bot_prompt.strip():
            raise HTTPException(status_code=400, detail="bot_prompt is required")
        
        if IS_PRODUCTION and request.bot_function == "interview_bot":
            config = _get_fly_config()
            fly_headers = _get_fly_headers(config["api_key"])
            current_region = os.getenv("FLY_REGION","sjc")

            logger.info(f"Spawning Fly Machine in region: {current_region}")

            logger.info(f"ENV: {os.getenv('FLY_REGION','No Region Found')}")

            FLY_API_KEY = config["api_key"]
            FLY_API_HOST = config["api_host"]
            FLY_APP_NAME = config["app_name"]

            FLY_HEADERS = fly_headers

            url=f"{FLY_API_HOST}/apps/{FLY_APP_NAME}/machines"
            headers = FLY_HEADERS

            async with httpx.AsyncClient(timeout=10.0) as client:
                #Get the image from existing Machine
                try:
                    res = await client.get(
                        url=url,
                        headers=headers
                        )
                    if res.status_code != 200:
                        raise Exception(f"Unable to get machine info from Fly: {res.text}")
                    
                    machines = res.json()
                    
                    if not machines:
                        raise Exception("No machines found")
                    
                    image = machines[0]['config']['image']

                except Exception as e:
                    logger.error(f"Failed to get machine image: {e}")
                    raise HTTPException(status_code=500, detail="Failed to retrieve machine configuration")
                
                #Create new machine for per User
                payload = {
                    "config": { 
                        "image": image,
                        "region": current_region,
                        "auto_destroy": True,
                        "guest": {"cpu_kind": "performance", "cpus": 1, "memory_mb": 2048},
                        "env": {
                            "RUN_MODE": "BOT_WORKER",
                            "BOT_ROOM_URL": request.daily_room_url.strip(),
                            "BOT_ROOM_TOKEN": request.daily_room_token or "",
                            "BOT_PROMPT": request.bot_prompt.strip(),
                            "BOT_NAME": request.bot_name.strip() or "Celia",
                            "BOT_STOP_PHRASE": request.stop_phrase or "",
                            "CALLBACK_URL": request.bubble_webhook_callback_url or "",
                            "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
                            "BOT_FUNCTION" : request.bot_function
                        },
                        "restart": {"policy": "no"} # Bot dies, machine stays dead
                    }
                }

                try:
                
                    response = await client.post(url, headers=headers, json=payload)

                    if response.status_code != 200:
                        logger.error(f"Fly Machine Error: {response.text}")
                        raise HTTPException(status_code=500, detail="Failed to spawn bot worker")
                    
                    # Wait for the machine to enter the started state
                    vm_id = response.json()['id']

                except Exception as e:
                    logger.error(f"Failed to create machine: {e}")
                    raise HTTPException(status_code=500, detail="Failed to spawn bot worker")
                
                # Wait for Machine to Start

                try:
                    res = await client.get(
                        url=url,
                        headers=headers,
                        timeout=30
                    )

                    if res.status_code != 200:
                        raise Exception(f"Bot was unable to enter started state: {res.text}")
                    
                    logger.info(f"Machine {vm_id} started successfully")

                except httpx.TimeoutException:
                    logger.error(f"Machine startup timeout for {vm_id}")
                    # raise HTTPException(status_code=500, detail="Machine startup timeout")

                except Exception as e:
                    logger.error(f"Failed to wait for machine startup: {e}")
                    raise HTTPException(status_code=500, detail="Failed to start bot worker")
                
        
        elif IS_DEVELOPMENT or request.bot_function == "disclaimer_bot":
            # Development Mode - Run inline
            bot = InterviewBot(
            daily_room_url=request.daily_room_url.strip(),
            daily_room_token=request.daily_room_token,
            bot_prompt=request.bot_prompt.strip(),
            bot_name=request.bot_name.strip(),
            stop_phrase=request.stop_phrase,
            bubble_webhook_callback_url=request.bubble_webhook_callback_url.strip(),
            bot_function=request.bot_function
        )

            background_tasks.add_task(bot.run)

        return InterviewResponse(status="success", message="Bot is joining the room")
    except Exception as e:
        logger.error(f"Error starting interview: {e}")
        raise HTTPException(status_code=500, detail=str(e))

        

# @app.post('/start-interview', response_model=InterviewResponse)
# async def start_interview(request:InterviewRequest,background_tasks:BackgroundTasks):
#     try:
#         bot = InterviewBot(
#             daily_room_url=request.daily_room_url,
#             daily_room_token=request.daily_room_token,
#             bot_prompt=request.bot_prompt,
#             bot_name=request.bot_name,
#             stop_phrase=request.stop_phrase,
#             bubble_webhook_callback_url=request.bubble_webhook_callback_url
#         )

        

#         background_tasks.add_task(bot.run)
        
#         return InterviewResponse(
#             status="started",
#             message=f"Interview bot joining meeting for {request.daily_room_url}"
#         )
        
#     except Exception as e:
#         logger.error(f"Error starting interview: {e}")
#         raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # import uvicorn
    # port = int(os.getenv("PORT", 7860))
    # logger.info(f"Starting server on port {port}")  # Add this line
    # logger.info(f"PORT environment variable: {os.getenv('PORT')}")  # Add this line
    # uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)

    if os.getenv("RUN_MODE") == "BOT_WORKER":
        import asyncio
        asyncio.run(run_bot_worker())
    else:
        import uvicorn
        port = int(os.getenv("PORT", 7860))
        uvicorn.run("main:app", host="0.0.0.0", port=port)
