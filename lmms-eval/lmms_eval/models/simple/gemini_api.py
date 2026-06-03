import base64
import io
import json
import math
import os
import pathlib
import re
import time
from typing import List, Tuple

from accelerate import Accelerator, DistributedType
from loguru import logger as eval_logger
from PIL import Image
import requests
from tqdm import tqdm

from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model

try:
    import google.generativeai as genai
    from google.generativeai.types import HarmBlockThreshold, HarmCategory

    NUM_SECONDS_TO_SLEEP = 30
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    genai.configure(api_key=GOOGLE_API_KEY)

except Exception as e:
    eval_logger.error(f"Error importing generativeai: {str(e)}")
    genai = None

try:
    import soundfile as sf
except Exception as e:
    eval_logger.warning(f"Error importing soundfile, audio generation will not work: {str(e)}")

try:
    from decord import VideoReader, cpu
except Exception as e:
    eval_logger.warning(f"Error importing decord, local video frame sampling will not work: {str(e)}")
    VideoReader = None
    cpu = None


@register_model("gemini_api")
class GeminiAPI(lmms):
    _MEDIA_RESOLUTION_ALIASES = {
        "UNSPECIFIED": "MEDIA_RESOLUTION_UNSPECIFIED",
        "MEDIA_RESOLUTION_UNSPECIFIED": "MEDIA_RESOLUTION_UNSPECIFIED",
        "LOW": "MEDIA_RESOLUTION_LOW",
        "MEDIA_RESOLUTION_LOW": "MEDIA_RESOLUTION_LOW",
        "MEDIUM": "MEDIA_RESOLUTION_MEDIUM",
        "MEDIA_RESOLUTION_MEDIUM": "MEDIA_RESOLUTION_MEDIUM",
        "HIGH": "MEDIA_RESOLUTION_HIGH",
        "MEDIA_RESOLUTION_HIGH": "MEDIA_RESOLUTION_HIGH",
        "ULTRA_HIGH": "MEDIA_RESOLUTION_ULTRA_HIGH",
        "ULTRAHIGH": "MEDIA_RESOLUTION_ULTRA_HIGH",
        "MEDIA_RESOLUTION_ULTRA_HIGH": "MEDIA_RESOLUTION_ULTRA_HIGH",
    }

    def __init__(
        self,
        model_version: str = "gemini-1.5-pro",
        # modality: str = "image",
        timeout: int = 120,
        api_transport: str = "sdk",
        thinking_level: str = None,
        continual_mode: bool = True,
        response_persistent_folder: str = "./logs/gemini_persistent_folder",
        interleave: bool = False,
        video_fps: float = None,
        max_video_frames: int = None,
        frame_max_edge: int = 768,
        media_resolution: str = None,
        # We will cache the Gemini API response in this path and use it for future requests
        **kwargs,
    ) -> None:
        super().__init__()
        self.model_version = model_version
        self.timeout = int(timeout)
        self.api_transport = (api_transport or "sdk").strip().lower()
        self.thinking_level = (thinking_level or "").strip().lower() or None
        self.model = None
        if self.api_transport == "sdk":
            self.model = genai.GenerativeModel(model_version)
        self.continual_mode = continual_mode
        self.response_persistent_file = ""
        self.interleave = interleave
        self.media_resolution = self._normalize_media_resolution(media_resolution)
        # if self.continual_mode and response_persistent_folder is None:
        #     raise ValueError("Continual mode requires a persistent path for the response. We will cache the Gemini API response in this path and use it for future requests. Please provide a valid path.")
        if self.continual_mode:
            self.response_persistent_folder = response_persistent_folder
            if not os.path.exists(self.response_persistent_folder):
                os.makedirs(self.response_persistent_folder)
            self.response_persistent_file = os.path.join(self.response_persistent_folder, f"{self.model_version}_response.json")

        if os.path.exists(self.response_persistent_file):
            with open(self.response_persistent_file, "r") as f:
                self.response_cache = json.load(f)
            self.cache_mode = "resume"
        else:
            self.response_cache = {}
            self.cache_mode = "start"

        accelerator = Accelerator()
        if accelerator.num_processes > 1:
            assert self.continual_mode is False, "Continual mode is not supported with distributed inference."
            assert accelerator.distributed_type in [DistributedType.FSDP, DistributedType.MULTI_GPU, DistributedType.DEEPSPEED], "Unsupported distributed type provided. Only DDP and FSDP are supported."
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            self.accelerator = accelerator
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes

        self.device = self.accelerator.device

        # self.modality = modality

        self.video_pool = []
        self.video_fps = float(video_fps) if video_fps is not None else None
        self.max_video_frames = int(max_video_frames) if max_video_frames is not None else None
        self.frame_max_edge = int(frame_max_edge) if frame_max_edge is not None else None
        if self.media_resolution is not None and self.api_transport != "rest":
            eval_logger.warning(
                "gemini_api media_resolution={} is only applied on the REST transport in this wrapper; current api_transport={} will ignore it",
                self.media_resolution,
                self.api_transport,
            )

    @classmethod
    def _normalize_media_resolution(cls, value):
        if value is None:
            return None
        normalized = str(value).strip()
        if not normalized:
            return None
        normalized = normalized.replace("-", "_").replace(" ", "_").upper()
        try:
            return cls._MEDIA_RESOLUTION_ALIASES[normalized]
        except KeyError as exc:
            raise ValueError(
                f"Unsupported media_resolution={value!r}; expected one of low/medium/high/ultra_high/unspecified or the MEDIA_RESOLUTION_* constants"
            ) from exc

    def free_video(self):
        for video in self.video_pool:
            video.delete()
        self.video_pool = []

    def _rest_url(self):
        api_key = os.getenv("GOOGLE_API_KEY") or GOOGLE_API_KEY
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY is required for Gemini REST requests")
        return f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_version}:generateContent?key={api_key}"

    @staticmethod
    def _pil_to_inline_data(image):
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return {
            "inline_data": {
                "mime_type": "image/png",
                "data": base64.b64encode(buf.getvalue()).decode("utf-8"),
            }
        }

    def _message_to_rest_parts(self, message):
        parts = []
        for item in message:
            if item is None:
                continue
            if isinstance(item, str):
                parts.append({"text": item})
            elif isinstance(item, Image.Image):
                parts.append(self._pil_to_inline_data(item))
            else:
                raise TypeError(f"Unsupported Gemini REST content type: {type(item)!r}")
        return parts

    def _rest_generate_content(self, message, gen_kwargs):
        generation_config = {
            "maxOutputTokens": gen_kwargs["max_new_tokens"],
            "temperature": gen_kwargs["temperature"],
        }
        if self.media_resolution is not None:
            generation_config["mediaResolution"] = self.media_resolution
        if self.thinking_level:
            generation_config["thinkingConfig"] = {"thinkingLevel": self.thinking_level}

        payload = {
            "contents": [{"parts": self._message_to_rest_parts(message)}],
            "generationConfig": generation_config,
        }
        parts = payload["contents"][0]["parts"]
        image_part_count = sum(1 for part in parts if isinstance(part, dict) and "inline_data" in part)
        text_part_count = sum(1 for part in parts if isinstance(part, dict) and "text" in part)

        response = requests.post(
            self._rest_url(),
            json=payload,
            timeout=self.timeout,
        )
        if response.status_code >= 400:
            body_excerpt = response.text[:4000] if response.text else ""
            eval_logger.error(
                "Gemini REST HTTP {} for model={} image_parts={} text_parts={} media_resolution={} body={}",
                response.status_code,
                self.model_version,
                image_part_count,
                text_part_count,
                self.media_resolution,
                body_excerpt,
            )
        response.raise_for_status()
        body = response.json()

        candidates = body.get("candidates") or []
        if not candidates:
            prompt_feedback = body.get("promptFeedback")
            eval_logger.warning("Gemini REST returned no candidates for {}: {}", self.model_version, prompt_feedback)
            return ""

        parts = candidates[0].get("content", {}).get("parts", [])
        texts = []
        for part in parts:
            text = part.get("text")
            if text:
                texts.append(text)
        return "".join(texts)

    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    def get_image_size(self, image):
        # Create a BytesIO object to store the image bytes
        img_byte_array = io.BytesIO()

        # Save the image to the BytesIO object
        image.save(img_byte_array, format="PNG")

        # Get the size of the BytesIO object
        img_size = img_byte_array.tell()

        return img_size

    def encode_video(self, video_path):
        uploaded_obj = genai.upload_file(path=video_path)
        time.sleep(5)
        self.video_pool.append(uploaded_obj)
        return uploaded_obj

    @staticmethod
    def _uniform_subsample_frame_indices(frame_indices, max_frames):
        if max_frames is None or max_frames <= 0 or len(frame_indices) <= max_frames:
            return frame_indices
        if max_frames == 1:
            return [frame_indices[len(frame_indices) // 2]]

        last_pos = len(frame_indices) - 1
        return [frame_indices[int(round(i * last_pos / (max_frames - 1)))] for i in range(max_frames)]

    def encode_video_as_frames(self, video_path):
        if VideoReader is None or cpu is None:
            raise RuntimeError("decord is required for local FPS-based video sampling but is not available")

        vr = VideoReader(video_path, ctx=cpu(0))
        total_frames = len(vr)
        if total_frames <= 0:
            raise ValueError(f"Video has no frames: {video_path}")

        if self.video_fps is None or self.video_fps <= 0:
            frame_indices = [0]
        else:
            native_fps = float(vr.get_avg_fps() or 0.0)
            if native_fps <= 0:
                native_fps = 1.0
            step = max(native_fps / self.video_fps, 1.0)
            frame_indices = []
            cursor = 0.0
            while int(cursor) < total_frames:
                frame_idx = int(math.floor(cursor))
                if not frame_indices or frame_idx != frame_indices[-1]:
                    frame_indices.append(frame_idx)
                cursor += step

        if not frame_indices:
            frame_indices = [0]

        fps_candidate_count = len(frame_indices)
        frame_indices = self._uniform_subsample_frame_indices(frame_indices, self.max_video_frames)

        frames = vr.get_batch(frame_indices).asnumpy()
        pil_frames = []
        for frame in frames:
            image = Image.fromarray(frame)
            if self.frame_max_edge is not None and self.frame_max_edge > 0:
                image.thumbnail((self.frame_max_edge, self.frame_max_edge))
            pil_frames.append(image)
        eval_logger.info(
            "Gemini local video sampling enabled for {}: sampled {} frames from {} fps candidates at fps={} (max_video_frames={})",
            video_path,
            len(pil_frames),
            fps_candidate_count,
            self.video_fps,
            self.max_video_frames,
        )
        return pil_frames

    def encode_audio(self, audio):
        audio_io = io.BytesIO()
        sf.write(audio_io, audio["array"], audio["sampling_rate"], format="WAV")
        return genai.upload_file(audio_io, mime_type="audio/wav")

    def convert_modality(self, images):
        converted = []
        for img in images:
            if isinstance(img, dict) and "sampling_rate" in img:  # audio
                audio = self.encode_audio(img)
                converted.append(audio)
            elif isinstance(img, str):  # video
                try:
                    if self.video_fps is not None:
                        converted.extend(self.encode_video_as_frames(img))
                    else:
                        converted.append(self.encode_video(img))
                except Exception as e:
                    eval_logger.error(f"Error converting video: {str(e)}")
            else:
                converted.append(img)
        return converted

    def construct_interleaved_input(self, content, media):
        pattern = r"<media_(\d+)>"
        parts = re.split(pattern, content)
        result = []
        for i, part in enumerate(parts):
            if i % 2 == 0:
                if part == "":
                    continue
                result.append(part)
            else:
                result.append(media[int(part)])

        return result

    def generate_until(self, requests) -> List[str]:
        res = []
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")

        def get_uuid(task, split, doc_id):
            return f"{task}___{split}___{doc_id}"

        for contexts, gen_kwargs, doc_to_visual, doc_id, task, split in [reg.args for reg in requests]:
            if self.continual_mode and self.cache_mode == "resume":
                doc_uuid = get_uuid(task, split, doc_id)
                if doc_uuid in self.response_cache:
                    content = self.response_cache[doc_uuid]
                    if content:
                        res.append(content)
                        pbar.update(1)
                        continue

            if "max_new_tokens" not in gen_kwargs:
                gen_kwargs["max_new_tokens"] = 1024
            if "temperature" not in gen_kwargs:
                gen_kwargs["temperature"] = 0

            config = genai.GenerationConfig(
                max_output_tokens=gen_kwargs["max_new_tokens"],
                temperature=gen_kwargs["temperature"],
            )

            visuals = [doc_to_visual(self.task_dict[task][split][doc_id])]
            visuals = self.flatten(visuals)
            visuals = self.convert_modality(visuals)

            if self.interleave:
                message = self.construct_interleaved_input(contexts, visuals)
            else:
                if self.video_fps is not None:
                    contexts = (
                        f"{contexts}\n\n"
                        f"Note: The visual input is provided as chronologically ordered frames sampled from the video at {self.video_fps} FPS."
                    )
                message = [contexts] + visuals

            for attempt in range(5):
                try:
                    if self.api_transport == "rest":
                        content = self._rest_generate_content(message, gen_kwargs)
                    else:
                        content = self.model.generate_content(
                            message,
                            generation_config=config,
                            safety_settings={
                                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                            },
                        )
                        content = content.text
                    break
                except Exception as e:
                    eval_logger.info(f"Attempt {attempt + 1} failed with error: {str(e)}")
                    if isinstance(e, ValueError):
                        try:
                            eval_logger.info(f"Prompt feed_back: {content.prompt_feedback}")
                            content = ""
                            break
                        except Exception:
                            pass
                    if attempt < 5 - 1:  # If we have retries left, sleep and then continue to next attempt
                        time.sleep(NUM_SECONDS_TO_SLEEP)
                    else:  # If this was the last attempt, log and return empty
                        eval_logger.error(f"All 5 attempts failed. Last error message: {str(e)}")
                        content = ""
            res.append(content)
            pbar.update(1)

            self.free_video()

            if self.continual_mode is True:  # Cache the response
                doc_uuid = get_uuid(task, split, doc_id)
                self.response_cache[doc_uuid] = content
                with open(self.response_persistent_file, "w") as f:
                    json.dump(self.response_cache, f)

        pbar.close()
        return res

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round generation for Gemini API")

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        # TODO
        assert False, "Gemini API not support"

    def get_image_audio_text_interleaved_messsage(self, image_path, audio_path, question):
        # image_path for list of image path
        # audio_path for list of audio path
        # question for question

        # fixed image token and no audio in text
        for index in range(1, 1 + len(image_path)):
            question = question.replace(f"[img{index}]", "<image>")
        for index in range(1, 1 + len(audio_path)):
            question = question.replace(f"[audio{index}]", "<audio>")

        text = question

        info_list = []
        image_counter = 0
        audio_counter = 0
        for part in re.split(r"(<image>|<audio>)", text):
            if part == "<image>":
                info_list.append(Image.open(image_path[image_counter]))
                image_counter += 1
            elif part == "<audio>":
                info_list.append({"mime_type": "audio/wav", "data": pathlib.Path(audio_path[audio_counter]).read_bytes()})
                audio_counter += 1
            else:
                if part == " ":
                    continue
                info_list.append(part)

        return info_list

    def get_video_audio_text_interleaved_message(self, video_path, audio_path, question):
        # image_path for list of image path
        # audio_path for list of audio path
        # question for question

        # fixed video token and no audio in text
        for index in range(1, 1 + len(video_path)):
            question = question.replace(f"[video{index}]", "<video>")
        for index in range(1, 1 + len(audio_path)):
            question = question.replace(f"[audio{index}]", "<audio>")

        text = question

        info_list = []
        video_counter = 0
        audio_counter = 0
        for part in re.split(r"(<video>|<audio>)", text):
            if part == "<video>":
                current_video_file_name = video_path[video_counter]
                current_video_file = genai.upload_file(path=current_video_file_name)
                while current_video_file.state.name == "processing":
                    print("uploading file")
                    time.sleep(5)
                    current_video_file = genai.get_file(current_video_file.name)
                if current_video_file.state.name == "FAILED":
                    print("uploading file failed, next question")
                    return 0
                info_list.append(current_video_file)
                video_counter += 1
            elif part == "<audio>":
                info_list.append({"mime_type": "audio/wav", "data": pathlib.Path(audio_path[audio_counter]).read_bytes()})
                audio_counter += 1
            else:
                if part == " ":
                    continue
                info_list.append(part)

        return info_list
