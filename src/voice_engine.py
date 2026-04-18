import speech_recognition as sr
import threading
import os
import tempfile
import asyncio
import edge_tts
import pygame
import hashlib
from src.config import Config
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class VoiceEngine:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self._tts_lock = threading.Lock()
        self._mic_ready = False  # Set to True once ambient calibration completes
        self._is_speaking = False
        
        # --- Tuning: prevent cutting off mid-command ---
        # Wait 1.5s of silence before deciding the user stopped talking (default 0.8s)
        self.recognizer.pause_threshold = 1.5
        # Ignore audio shorter than 0.5s (noise bursts, not real speech)
        self.recognizer.phrase_threshold = 0.5
        # Minimum silence at the end to finalize a phrase
        self.recognizer.non_speaking_duration = 0.8
        
        # Initialize pygame mixer for audio playback
        pygame.mixer.init()
        
        # Persistent event loop for async TTS (avoids creating new loop per speak())
        self._loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(target=self._run_event_loop, daemon=True)
        self._loop_thread.start()
        
        # TTS audio cache for common phrases
        self._tts_cache_dir = os.path.join(tempfile.gettempdir(), "assistant_tts_cache")
        os.makedirs(self._tts_cache_dir, exist_ok=True)
        
        # Calibrate ambient noise in background so the GUI opens instantly
        threading.Thread(target=self._calibrate_mic, daemon=True).start()
        
        logger.info("Voice engine initialized with edge-tts, pygame, and persistent event loop")
    
    def _run_event_loop(self):
        """Run the persistent event loop in its own thread."""
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()
    
    def _calibrate_mic(self):
        """Calibrate ambient noise once in background — does not block startup."""
        try:
            with sr.Microphone() as source:
                logger.info("Calibrating ambient noise (background)...")
                self.recognizer.adjust_for_ambient_noise(source, duration=2)
                # Lower minimum energy threshold to catch quieter speech
                self.recognizer.energy_threshold = max(self.recognizer.energy_threshold, 150)
                # Lock energy threshold so it doesn't drift during speech
                self.recognizer.dynamic_energy_threshold = False
                logger.info(f"Ambient noise calibrated. Energy threshold: {self.recognizer.energy_threshold}")
        except Exception as e:
            logger.warning(f"Microphone calibration failed: {e}")
            self.recognizer.energy_threshold = 300
            self.recognizer.dynamic_energy_threshold = False
        finally:
            self._mic_ready = True
    
    def listen(self, timeout=None, phrase_time_limit=None, retries=None):
        """
        Listen for voice input with retry logic.
        Ambient noise is pre-calibrated at startup — no delay per call.
        
        Args:
            timeout: Seconds to wait for speech to start (default from config)
            phrase_time_limit: Max seconds for a phrase (default from config)
            retries: Number of retry attempts (default from config)
            
        Returns:
            Recognized text or None/error string
        """
        timeout = timeout or Config.LISTEN_TIMEOUT
        phrase_time_limit = phrase_time_limit or Config.PHRASE_TIME_LIMIT
        retries = retries or Config.LISTEN_RETRIES
        
        for attempt in range(retries):
            try:
                with sr.Microphone() as source:
                    # No ambient noise calibration here — done once in __init__
                    logger.info(f"Listening... (attempt {attempt + 1}/{retries})")
                    
                    audio = self.recognizer.listen(
                        source,
                        timeout=timeout,
                        phrase_time_limit=phrase_time_limit
                    )
                    
                    text = self.recognizer.recognize_google(audio)
                    logger.info(f"Recognized: {text}")
                    return text
                    
            except sr.WaitTimeoutError:
                logger.warning(f"Timeout waiting for speech (attempt {attempt + 1}/{retries})")
                if attempt < retries - 1:
                    continue
                return "timeout"
                
            except sr.UnknownValueError:
                logger.warning(f"Could not understand audio (attempt {attempt + 1}/{retries})")
                if attempt < retries - 1:
                    continue
                return None
                
            except sr.RequestError as e:
                logger.error(f"Speech recognition API error: {e}")
                return "API unavailable"
                
            except Exception as e:
                logger.error(f"Unexpected error during listening: {e}", exc_info=True)
                return None
        
        return None

    def _get_cache_path(self, text):
        """Get the cache file path for a given text string."""
        # Create a hash of the text + voice settings to use as filename
        key = f"{text}|{Config.VOICE_NAME}|{Config.VOICE_RATE}"
        hash_key = hashlib.md5(key.encode()).hexdigest()
        return os.path.join(self._tts_cache_dir, f"{hash_key}.mp3")

    def speak(self, text, callback=None):
        """Speak the given text (non-blocking — runs in a separate thread)"""
        # Stop any currently playing audio so a new one can start immediately
        self.stop()
        thread = threading.Thread(target=self._speak_sync, args=(text, callback), daemon=True)
        thread.start()
        
    async def _generate_audio(self, text, output_file):
        """Generate audio using edge_tts asynchronously"""
        voice = Config.VOICE_NAME
        rate = Config.VOICE_RATE
        volume = Config.VOICE_VOLUME
        pitch = Config.VOICE_PITCH
        
        communicate = edge_tts.Communicate(text, voice, rate=rate, volume=volume, pitch=pitch)
        await communicate.save(output_file)
    
    def _speak_sync(self, text, callback=None):
        """Internal synchronous speak — called from thread"""
        with self._tts_lock:
            try:
                logger.debug(f"Speaking: {text[:50]}...")
                self._is_speaking = True
                
                # Check cache first
                cache_path = self._get_cache_path(text)
                
                if os.path.exists(cache_path):
                    # Cache hit — use pre-generated audio
                    temp_filename = cache_path
                    is_cached = True
                    logger.debug(f"TTS cache hit: {cache_path}")
                else:
                    # Cache miss — generate new audio
                    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
                    temp_filename = temp_file.name
                    temp_file.close()
                    is_cached = False
                    
                    # Generate audio using the persistent event loop (faster than asyncio.run())
                    future = asyncio.run_coroutine_threadsafe(
                        self._generate_audio(text, temp_filename),
                        self._loop
                    )
                    future.result(timeout=30)  # Wait up to 30s for TTS generation
                    
                    # Cache short phrases for instant replay next time
                    if len(text) < 200:
                        try:
                            import shutil
                            shutil.copy2(temp_filename, cache_path)
                            logger.debug(f"TTS cached: {cache_path}")
                        except Exception:
                            pass
                
                if not self._is_speaking:
                    # Stop was called during generation
                    if not is_cached:
                        try:
                            os.remove(temp_filename)
                        except Exception:
                            pass
                    return
                
                if not pygame.mixer.get_init():
                    pygame.mixer.init()
                    
                # Play audio using pygame
                pygame.mixer.music.load(temp_filename)
                pygame.mixer.music.play()
                
                # Wait for playback to finish
                while pygame.mixer.music.get_busy() and self._is_speaking:
                    pygame.time.Clock().tick(10)
                    
                # Clean up pygame and file
                if hasattr(pygame.mixer.music, 'unload'):
                    pygame.mixer.music.unload()
                else:
                    pygame.mixer.music.stop()
                    
                # Only delete temp files, not cached ones
                if not is_cached:
                    try:
                        os.remove(temp_filename)
                    except Exception as e:
                        logger.debug(f"Could not remove temp audio file: {e}")
                    
            except Exception as e:
                logger.error(f"Failed to speak: {e}")
            finally:
                self._is_speaking = False
                if callback:
                    callback()
    
    def stop(self):
        """Stop TTS immediately"""
        try:
            self._is_speaking = False
            if pygame.mixer.get_init() and pygame.mixer.music.get_busy():
                pygame.mixer.music.stop()
                if hasattr(pygame.mixer.music, 'unload'):
                    pygame.mixer.music.unload()
        except Exception as e:
            logger.error(f"Failed to stop TTS: {e}")
