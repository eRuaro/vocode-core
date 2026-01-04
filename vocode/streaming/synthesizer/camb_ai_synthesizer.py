import asyncio
import audioop
import hashlib
from typing import Optional

from loguru import logger

from vocode import getenv
from vocode.streaming.models.audio import AudioEncoding, SamplingRate
from vocode.streaming.models.message import BaseMessage
from vocode.streaming.models.synthesizer import CambAiSynthesizerConfig
from vocode.streaming.synthesizer.base_synthesizer import BaseSynthesizer, SynthesisResult
from vocode.streaming.utils.create_task import asyncio_create_task

CAMB_AI_BASE_URL = "https://client.camb.ai/apis"
CAMB_AI_NATIVE_SAMPLE_RATE = 24000


class CambAiException(Exception):
    pass


class CambAiSynthesizer(BaseSynthesizer[CambAiSynthesizerConfig]):
    def __init__(
        self,
        synthesizer_config: CambAiSynthesizerConfig,
    ):
        super().__init__(synthesizer_config)

        self.api_key = synthesizer_config.api_key or getenv("CAMB_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Camb.ai API key is required. Set CAMB_API_KEY environment variable "
                "or pass api_key in CambAiSynthesizerConfig."
            )

        self.voice_id = synthesizer_config.voice_id
        self.model_id = synthesizer_config.model_id
        self.language = synthesizer_config.language
        self.speed = synthesizer_config.speed
        self.user_instructions = synthesizer_config.user_instructions
        self.words_per_minute = 150

        self.needs_resample = False
        self.target_sample_rate = self.synthesizer_config.sampling_rate

        if self.synthesizer_config.audio_encoding == AudioEncoding.LINEAR16:
            self.output_format = "pcm_s16le"
            if self.synthesizer_config.sampling_rate != CAMB_AI_NATIVE_SAMPLE_RATE:
                self.needs_resample = True
        elif self.synthesizer_config.audio_encoding == AudioEncoding.MULAW:
            self.output_format = "pcm_s16le"
            self.needs_resample = True
        else:
            raise ValueError(
                f"Unsupported audio encoding: {self.synthesizer_config.audio_encoding}"
            )

    async def create_speech_uncached(
        self,
        message: BaseMessage,
        chunk_size: int,
        is_first_text_chunk: bool = False,
        is_sole_text_chunk: bool = False,
    ) -> SynthesisResult:
        text = message.text
        if len(text) < 3:
            text = text + "   "
        if len(text) > 3000:
            logger.warning(
                f"Text length {len(text)} exceeds Camb.ai limit of 3000 characters. Truncating."
            )
            text = text[:3000]

        self.total_chars += len(message.text)

        url = f"{CAMB_AI_BASE_URL}/tts-stream"
        headers = {
            "x-api-key": self.api_key,
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

        body = {
            "text": text,
            "voice_id": self.voice_id,
            "language": self.language,
            "speech_model": self.model_id,
            "output_configuration": {
                "format": self.output_format,
            },
            "voice_settings": {
                "speed": self.speed,
            },
        }

        if self.user_instructions and self.model_id == "mars-8-instruct":
            body["user_instructions"] = self.user_instructions

        chunk_queue: asyncio.Queue[Optional[bytes]] = asyncio.Queue()
        asyncio_create_task(
            self.get_chunks(url, headers, body, chunk_size, chunk_queue),
        )

        return SynthesisResult(
            self.chunk_result_generator_from_queue(chunk_queue),
            lambda seconds: self.get_message_cutoff_from_voice_speed(message, seconds, 150),
        )

    @classmethod
    def get_voice_identifier(cls, synthesizer_config: CambAiSynthesizerConfig) -> str:
        hashed_api_key = hashlib.sha256(f"{synthesizer_config.api_key}".encode("utf-8")).hexdigest()
        return ":".join(
            (
                "camb_ai",
                hashed_api_key,
                str(synthesizer_config.voice_id),
                str(synthesizer_config.model_id),
                str(synthesizer_config.language),
                str(synthesizer_config.speed),
                synthesizer_config.audio_encoding,
            )
        )

    async def get_chunks(
        self,
        url: str,
        headers: dict,
        body: dict,
        chunk_size: int,
        chunk_queue: asyncio.Queue[Optional[bytes]],
    ):
        try:
            async_client = self.async_requestor.get_client()
            stream = await async_client.send(
                async_client.build_request(
                    "POST",
                    url,
                    headers=headers,
                    json=body,
                    timeout=60.0,
                ),
                stream=True,
            )

            if not stream.is_success:
                error = await stream.aread()
                error_message = error.decode("utf-8")
                status_code = stream.status_code

                if status_code == 401:
                    raise CambAiException(
                        "Invalid Camb.ai API key. Set CAMB_API_KEY environment variable "
                        "with your API key from https://camb.ai"
                    )
                elif status_code == 403:
                    raise CambAiException(
                        f"Voice ID {self.voice_id} is not accessible with your API key. "
                        "Use Camb.ai's list-voices endpoint to see available voices."
                    )
                elif status_code == 404:
                    raise CambAiException(f"Invalid voice ID: {self.voice_id}")
                elif status_code == 429:
                    raise CambAiException(
                        "Camb.ai rate limit exceeded. Please wait before making more requests."
                    )
                else:
                    raise CambAiException(
                        f"Camb.ai API returned {status_code} status code: {error_message}"
                    )

            async for chunk in stream.aiter_bytes(chunk_size):
                processed_chunk = chunk
                if self.needs_resample:
                    processed_chunk = self._resample_chunk(
                        chunk,
                        CAMB_AI_NATIVE_SAMPLE_RATE,
                        self.target_sample_rate,
                    )
                if self.synthesizer_config.audio_encoding == AudioEncoding.MULAW:
                    processed_chunk = audioop.lin2ulaw(processed_chunk, 2)
                chunk_queue.put_nowait(processed_chunk)
        except asyncio.CancelledError:
            pass
        finally:
            chunk_queue.put_nowait(None)
