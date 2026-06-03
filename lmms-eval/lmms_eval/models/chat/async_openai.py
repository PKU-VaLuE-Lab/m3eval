import asyncio
import copy
import json
import os
import shutil
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from multiprocessing import cpu_count
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse

from accelerate import Accelerator, DistributedType
from dotenv import load_dotenv
from loguru import logger as eval_logger
from openai import AsyncOpenAI
from tqdm import tqdm

from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.imports import optional_import
from lmms_eval.models.model_utils.concurrency_control import (
    AdaptiveConcurrencyConfig,
    decide_next_concurrency,
    extract_text_prefix_from_chat_messages,
    is_rate_limit_error,
    make_prefix_hash,
    parse_bool,
)
from lmms_eval.protocol import ChatMessages

VideoReader, _ = optional_import("decord", "VideoReader")
cpu, _ = optional_import("decord", "cpu")

load_dotenv(verbose=True)


def _parse_base_url_values(raw: object) -> List[str]:
    if raw is None:
        return []
    if isinstance(raw, (list, tuple)):
        items = [str(item).strip() for item in raw if str(item).strip()]
    else:
        text = str(raw).strip()
        if not text:
            return []
        if text.startswith("[") and text.endswith("]"):
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, list):
                items = [str(item).strip() for item in parsed if str(item).strip()]
            else:
                items = [item.strip() for item in text.split("|") if item.strip()]
        else:
            delimiter = "|" if "|" in text else ","
            items = [item.strip() for item in text.split(delimiter) if item.strip()]

    normalized: List[str] = []
    for item in items:
        if "%" in item:
            try:
                from urllib.parse import unquote

                item = unquote(item)
            except Exception:
                pass
        if item.endswith("/"):
            item = item.rstrip("/")
        if item:
            normalized.append(item)
    return normalized


def _parse_boolish(value: object) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    lowered = str(value).strip().lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off"}:
        return False
    if lowered in {"", "auto", "none", "null"}:
        return None
    raise ValueError(f"Invalid bool-like value: {value!r}")


def _is_local_base_url(base_url: str) -> bool:
    parsed = urlparse(str(base_url).strip())
    host = (parsed.hostname or "").strip().lower()
    return host in {"127.0.0.1", "localhost", "::1", "0.0.0.0"}


def _looks_like_qwen35_model(model_name: str) -> bool:
    lowered = str(model_name or "").strip().lower()
    return "qwen3.5" in lowered or "qwen35" in lowered


def _resolve_enable_thinking(
    *,
    base_url: str,
    model_version: str,
    enable_thinking: object,
    disable_thinking: object,
) -> Optional[bool]:
    enabled = _parse_boolish(enable_thinking)
    if enabled is not None:
        return enabled

    disabled = _parse_boolish(disable_thinking)
    if disabled is True:
        return False
    if disabled is False:
        if _is_local_base_url(base_url) and _looks_like_qwen35_model(model_version):
            return True
        return None

    if _is_local_base_url(base_url) and _looks_like_qwen35_model(model_version):
        # Local Qwen3.5 OpenAI-compatible backends default to thinking mode.
        # That often yields `content=None` plus a `reasoning` string, which
        # breaks strict answer-format tasks that expect the final answer in content.
        return False
    return None


def _resolve_request_extra_body(
    *,
    base_url: str,
    enable_thinking: Optional[bool],
    extra_body: object,
) -> dict:
    payload = copy.deepcopy(extra_body) if isinstance(extra_body, dict) else {}
    if enable_thinking is None:
        return payload
    if _is_local_base_url(base_url):
        chat_template_kwargs = payload.get("chat_template_kwargs")
        if not isinstance(chat_template_kwargs, dict):
            chat_template_kwargs = {}
        chat_template_kwargs["enable_thinking"] = bool(enable_thinking)
        payload["chat_template_kwargs"] = chat_template_kwargs
        return payload
    payload["enable_thinking"] = bool(enable_thinking)
    return payload


@dataclass
class _AdaptiveConcurrencyTracker:
    """Tracker for adaptive concurrency control statistics.

    Attributes:
        failed_requests: Number of requests that failed.
        rate_limited_requests: Number of requests that were rate-limited.
        latencies: List of request latencies in seconds.
        completed_count: Total number of completed requests since last update.
    """

    failed_requests: int = 0
    rate_limited_requests: int = 0
    latencies: List[float] = field(default_factory=list)
    completed_count: int = 0


@register_model("async_openai")
class AsyncOpenAIChat(lmms):
    is_simple = False

    def __init__(
        self,
        model_version: str = "grok-2-latest",
        base_url: str = None,
        base_urls: object = None,
        api_key: str = None,
        enable_thinking: object = None,
        disable_thinking: object = None,
        extra_body: object = None,
        timeout: int = 600,
        retry_backoff_s: Optional[float] = None,
        max_retries: int = 5,
        max_size_in_mb: int = 20,
        mcp_server_path: str = None,
        num_cpus: int = None,
        num_concurrent: Optional[int] = None,
        work_dir: str = None,
        fps: Optional[int] = None,
        nframes: Optional[int] = 64,
        max_frames: Optional[int] = 768,
        max_pixels: Optional[int] = 151200,
        min_pixels: Optional[int] = 28 * 28,
        adaptive_concurrency: bool = False,
        adaptive_min_concurrency: int = 1,
        adaptive_max_concurrency: int = 128,
        adaptive_target_latency_s: float = 15.0,
        adaptive_increase_step: float = 0.1,
        adaptive_decrease_factor: float = 0.7,
        adaptive_failure_threshold: float = 0.05,
        message_format: str = "openai",
        prefix_aware_queue: bool = True,
        prefix_hash_chars: int = 256,
        **kwargs,
    ) -> None:
        super().__init__()
        self.model_version = model_version
        self.timeout = timeout
        self.retry_backoff_s = max(0.0, float(1.0 if retry_backoff_s is None else retry_backoff_s))
        self.max_retries = max_retries
        self.max_size_in_mb = max_size_in_mb
        if num_cpus is None:
            self.num_cpus = cpu_count() // 2
        else:
            self.num_cpus = num_cpus
        self.work_dir = work_dir if work_dir is not None else tempfile.mkdtemp()
        self.fps = fps
        self.nframes = nframes
        raw_base_urls = base_urls if base_urls is not None else base_url
        if raw_base_urls is None:
            raw_base_urls = os.getenv("OPENAI_API_BASE_URLS") or os.getenv("OPENAI_API_BASE")
        self.base_urls = _parse_base_url_values(raw_base_urls)
        if not self.base_urls:
            raise ValueError("AsyncOpenAIChat requires at least one base_url/base_urls endpoint")
        self.base_url = self.base_urls[0]
        self.api_key = api_key if api_key is not None else os.getenv("OPENAI_API_KEY")
        self.clients = [AsyncOpenAI(api_key=self.api_key, base_url=item, timeout=timeout) for item in self.base_urls]
        self.client = self.clients[0]
        self._endpoint_request_counts = {item: 0 for item in self.base_urls}
        self.enable_thinking = _resolve_enable_thinking(
            base_url=self.base_url,
            model_version=self.model_version,
            enable_thinking=enable_thinking,
            disable_thinking=disable_thinking,
        )
        self.request_extra_body = _resolve_request_extra_body(
            base_url=self.base_url,
            enable_thinking=self.enable_thinking,
            extra_body=extra_body,
        )
        self.max_pixels = max_pixels
        self.min_pixels = min_pixels
        self.max_frames = max_frames
        self.num_concurrent = max(1, int(num_concurrent)) if num_concurrent is not None else max(1, self.num_cpus)
        self.adaptive_concurrency = parse_bool(adaptive_concurrency)
        self.adaptive_config = AdaptiveConcurrencyConfig.from_raw(
            min_concurrency=adaptive_min_concurrency,
            max_concurrency=adaptive_max_concurrency,
            target_latency_s=adaptive_target_latency_s,
            increase_step=adaptive_increase_step,
            decrease_factor=adaptive_decrease_factor,
            failure_threshold=adaptive_failure_threshold,
        )
        self.message_format = message_format
        self.prefix_aware_queue = parse_bool(prefix_aware_queue)
        self.prefix_hash_chars = max(32, int(prefix_hash_chars))
        if mcp_server_path is not None:
            from lmms_eval.mcp import MCPClient

            self.mcp_client = MCPClient(mcp_server_path)
            os.makedirs(self.work_dir, exist_ok=True)
        else:
            self.mcp_client = None

        accelerator = Accelerator()
        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [
                DistributedType.FSDP,
                DistributedType.MULTI_GPU,
                DistributedType.DEEPSPEED,
            ], "Unsupported distributed type provided. Only DDP and FSDP are supported."
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
        if self.accelerator.is_local_main_process:
            eval_logger.info(
                "AsyncOpenAI endpoints: {} | num_concurrent={} | enable_thinking={}",
                " | ".join(self.base_urls),
                self.num_concurrent,
                self.enable_thinking,
            )

    @property
    def model(self):
        return self.client

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        assert False, "TODO, not implemented"

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round generation for LLaVAHF")

    def _build_video_kwargs(self) -> Dict:
        video_kwargs = {"max_pixels": self.max_pixels, "min_pixels": self.min_pixels}
        if self.fps is not None:
            video_kwargs["fps"] = self.fps
        else:
            video_kwargs["nframes"] = self.nframes
        if self.max_frames is not None:
            video_kwargs["max_frames"] = self.max_frames
        return video_kwargs

    def prepare_messages(self, chat_messages: ChatMessages) -> Tuple[List[Dict], Dict]:
        """Prepare API-compatible messages from chat messages.

        Dispatches to the appropriate message format based on ``self.message_format``.
        Supported formats:
            - ``"default"`` (default): standard OpenAI vision messages.
            - ``"qwen3_vl"``: Qwen3-VL format with per-frame timestamps.

        Args:
            chat_messages: The chat messages object containing user queries and media.

        Returns:
            A tuple of (messages, video_kwargs) where messages is the API-compatible
            message format and video_kwargs contains video processing parameters.
        """
        video_kwargs = self._build_video_kwargs()
        if self.message_format == "qwen3_vl":
            messages = chat_messages.to_qwen3_vl_openai_messages(video_kwargs)
        else:
            messages = chat_messages.to_openai_messages(video_kwargs)
        return messages, video_kwargs

    def _get_initial_concurrency(self) -> int:
        """Get the initial concurrency level for request processing.

        Returns:
            Initial concurrency value based on CPU count and adaptive config.
        """
        if self.adaptive_concurrency:
            return min(self.num_concurrent, self.adaptive_config.max_concurrency)
        return self.num_concurrent

    def _client_for_request(self, idx: int) -> AsyncOpenAI:
        return self._client_and_endpoint_for_request(idx, attempt=0)[0]

    def _client_and_endpoint_for_request(self, idx: int, attempt: int) -> tuple[AsyncOpenAI, str]:
        endpoint_index = (int(idx) + int(attempt)) % len(self.clients)
        return self.clients[endpoint_index], self.base_urls[endpoint_index]

    @staticmethod
    def _coerce_message_content(message) -> str:  # noqa: ANN001
        content = getattr(message, "content", None)
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            text_chunks: list[str] = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text = item.get("text")
                    if isinstance(text, str) and text:
                        text_chunks.append(text)
            if text_chunks:
                return "\n".join(text_chunks)
        reasoning_content = getattr(message, "reasoning_content", None) or getattr(message, "reasoning", None)
        if isinstance(reasoning_content, str):
            return reasoning_content
        model_extra = getattr(message, "model_extra", None)
        if isinstance(model_extra, dict):
            extra_reasoning = model_extra.get("reasoning_content") or model_extra.get("reasoning")
            if isinstance(extra_reasoning, str):
                return extra_reasoning
        return ""

    def _compute_dispatch_order(self, requests: List[Instance]) -> List[int]:
        """Compute the dispatch order for requests.

        If prefix_aware_queue is enabled, requests are sorted by their text prefix hash
        to improve cache locality and performance.

        Args:
            requests: List of request instances.

        Returns:
            List of request indices in dispatch order.
        """
        dispatch_order = list(range(len(requests)))
        if not self.prefix_aware_queue:
            return dispatch_order
        prefix_hashes = {}
        for idx in dispatch_order:
            request_obj = requests[idx]
            prefix_text = request_obj.args[0] if isinstance(request_obj.args[0], str) else ""
            if not prefix_text:
                _, doc_to_messages, _, doc_id, task, split = request_obj.args
                chat_messages_raw = doc_to_messages(self.task_dict[task][split][doc_id])
                prefix_text = extract_text_prefix_from_chat_messages(chat_messages_raw, self.prefix_hash_chars)
            prefix_hashes[idx] = make_prefix_hash(prefix_text, self.prefix_hash_chars)
        dispatch_order.sort(key=lambda idx: (prefix_hashes[idx], idx))
        return dispatch_order

    async def _process_with_retry(self, req: Instance, idx: int) -> Tuple[str, int, bool, bool, float]:
        """Process a single request with retry logic.

        Args:
            req: The request instance to process.
            idx: The original index of the request.

        Returns:
            A tuple containing:
                - content: The response content or error message.
                - original_idx: The original request index.
                - success: Whether the request succeeded.
                - rate_limited: Whether the request was rate-limited.
                - elapsed: Time taken in seconds.
        """
        started_at = time.time()
        rate_limited = False
        last_error_msg = "unknown error"
        for attempt in range(self.max_retries):
            try:
                content, original_idx = await self.maybe_forward_with_tool(req, idx, attempt=attempt)
                elapsed = time.time() - started_at
                return content, original_idx, True, rate_limited, elapsed
            except Exception as exc:
                error_msg = str(exc)
                last_error_msg = error_msg
                rate_limited = rate_limited or is_rate_limit_error(error_msg)
                eval_logger.info(f"Attempt {attempt + 1}/{self.max_retries} failed for request {idx} with error: {error_msg}")
                if attempt == self.max_retries - 1:
                    eval_logger.error(f"All {self.max_retries} attempts failed. Last error: {error_msg}")
                else:
                    await asyncio.sleep(self.retry_backoff_s)
        elapsed = time.time() - started_at
        error_preview = last_error_msg.replace("\n", " ")[:200]
        failure_content = f"[LMMS_EVAL_REQUEST_FAILED after {self.max_retries} retries] {error_preview}"
        return failure_content, idx, False, rate_limited, elapsed

    def _should_update_concurrency(self, tracker: _AdaptiveConcurrencyTracker, force: bool) -> bool:
        """Determine if concurrency should be updated based on tracker state.

        Args:
            tracker: The concurrency statistics tracker.
            force: If True, bypass the sample threshold check.

        Returns:
            True if concurrency should be updated, False otherwise.
        """
        if not self.adaptive_concurrency:
            return False
        if force:
            return tracker.completed_count > 0
        sample_threshold = max(4, self._get_initial_concurrency())
        return tracker.completed_count >= sample_threshold

    def _update_concurrency(self, tracker: _AdaptiveConcurrencyTracker, current_concurrency: int, force: bool) -> int:
        """Update concurrency based on tracked statistics.

        Args:
            tracker: The concurrency statistics tracker.
            current_concurrency: The current concurrency level.
            force: If True, bypass the sample threshold check.

        Returns:
            The updated concurrency level.
        """
        if not self._should_update_concurrency(tracker, force):
            return current_concurrency
        if tracker.completed_count <= 0:
            return current_concurrency
        decision = decide_next_concurrency(
            current_concurrency=current_concurrency,
            total_requests=tracker.completed_count,
            failed_requests=tracker.failed_requests,
            rate_limited_requests=tracker.rate_limited_requests,
            latencies=tracker.latencies,
            config=self.adaptive_config,
        )
        if decision.next_concurrency != decision.current_concurrency:
            eval_logger.info(
                "Adaptive concurrency update: "
                f"{decision.current_concurrency} -> "
                f"{decision.next_concurrency} "
                f"(fail_rate={decision.failure_rate:.3f}, "
                f"rate_limit_rate={decision.rate_limit_rate:.3f}, "
                f"p95_latency={decision.p95_latency_s:.3f}s)"
            )
        tracker.failed_requests = 0
        tracker.rate_limited_requests = 0
        tracker.latencies = []
        tracker.completed_count = 0
        return decision.next_concurrency

    async def _run_scheduling_loop(
        self,
        requests: List[Instance],
        dispatch_order: List[int],
        pbar: tqdm,
        initial_concurrency: int,
    ) -> List[Tuple[str, int]]:
        """Run the main scheduling loop with concurrency control.

        Args:
            requests: List of request instances to process.
            dispatch_order: List of request indices in dispatch order.
            pbar: Progress bar for tracking completion.
            initial_concurrency: Initial concurrency level.

        Returns:
            List of (content, original_idx) tuples for completed requests.
        """
        current_concurrency = initial_concurrency
        tracker = _AdaptiveConcurrencyTracker()
        in_flight: dict[asyncio.Task, int] = {}
        cursor = 0
        res: List[Tuple[str, int]] = []

        while cursor < len(dispatch_order) or in_flight:
            while cursor < len(dispatch_order) and len(in_flight) < max(1, current_concurrency):
                request_index = dispatch_order[cursor]
                task = asyncio.create_task(self._process_with_retry(requests[request_index], request_index))
                in_flight[task] = request_index
                cursor += 1

            if not in_flight:
                break

            done, _ = await asyncio.wait(in_flight, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                in_flight.pop(task, None)
                (
                    content,
                    request_idx,
                    success,
                    rate_limited,
                    elapsed,
                ) = task.result()
                res.append((content, request_idx))
                if not success:
                    tracker.failed_requests += 1
                if rate_limited:
                    tracker.rate_limited_requests += 1
                tracker.latencies.append(elapsed)
                tracker.completed_count += 1
                pbar.update(1)
                current_concurrency = self._update_concurrency(tracker, current_concurrency, force=False)

        current_concurrency = self._update_concurrency(tracker, current_concurrency, force=True)
        return res

    async def maybe_forward_with_tool(self, request: Instance, idx: int, attempt: int = 0):
        """Forward request to OpenAI API, using tools if available.

        Handles chat messages and tool calls asynchronously. Retrieves messages,
        prepares payload, and sends to OpenAI API. If tool calls are needed,
        executes them via MCP client and continues until final response.

        Args:
            request: The request instance containing chat messages and parameters.
            idx: The index of the request in the batch.

        Returns:
            A tuple of (response_content, original_idx).
        """
        ctx, doc_to_messages, gen_kwargs, doc_id, task, split = request.args
        chat_messages = doc_to_messages(self.task_dict[task][split][doc_id])
        chat_messages: ChatMessages = ChatMessages(**{"messages": chat_messages})
        messages, video_kwargs = self.prepare_messages(chat_messages)
        images, videos, audios = chat_messages.extract_media()
        client, endpoint_url = self._client_and_endpoint_for_request(idx, attempt)
        self._endpoint_request_counts[endpoint_url] = self._endpoint_request_counts.get(endpoint_url, 0) + 1
        if self.mcp_client is not None:
            for image_idx, image in enumerate(images):
                image_path = os.path.join(self.work_dir, f"{uuid.uuid4()}.jpg")
                image.save(image_path)
                messages[-1]["content"].append(
                    {
                        "type": "text",
                        "text": f"\nImage {image_idx} has image path: {image_path}",
                    }
                )
            for video_idx, video in enumerate(videos):
                messages[-1]["content"].append(
                    {
                        "type": "text",
                        "text": f"\nVideo {video_idx} has video path: {video}",
                    }
                )

        payload = {"messages": messages}
        payload["model"] = self.model_version
        all_response = ""

        if "max_new_tokens" not in gen_kwargs:
            gen_kwargs["max_new_tokens"] = 1024
        if "temperature" not in gen_kwargs:
            gen_kwargs["temperature"] = 0
        if "top_p" not in gen_kwargs:
            gen_kwargs["top_p"] = None
        if "do_sample" not in gen_kwargs:
            gen_kwargs["do_sample"] = False
        payload["max_tokens"] = gen_kwargs["max_new_tokens"]
        payload["temperature"] = gen_kwargs["temperature"]
        if self.request_extra_body:
            payload["extra_body"] = copy.deepcopy(self.request_extra_body)

        if self.mcp_client is not None:
            functions = await self.mcp_client.get_function_list()
            payload["tools"] = functions
            payload["tool_choice"] = "auto"

        response = await client.chat.completions.create(**payload)
        last_response = self._coerce_message_content(response.choices[0].message)
        all_response += last_response

        while response.choices[0].finish_reason == "tool_calls":
            messages.append({"role": "assistant", "content": last_response})
            messages.append(
                {
                    "role": "assistant",
                    "tool_calls": response.choices[0].message.tool_calls,
                }
            )
            message = response.choices[0].message
            tool_messages = []
            if message.tool_calls:
                eval_logger.debug("Calling tool with MCP client")
                for call in message.tool_calls:
                    eval_logger.debug(f"Calling {call.function.name}...")
                    result = await self.mcp_client.run_tool(call.function.name, eval(call.function.arguments))
                    all_response += f"<tool_call>{call.function.name} {call.function.arguments}</tool_call><tool_response>"
                    tool_messages.append({"role": "tool", "name": call.function.name, "content": []})
                    for content in result.content:
                        tool_message = self.mcp_client.convert_result_to_openai_format(content)
                        for content in tool_message:
                            if content["type"] == "image_url":
                                all_response += "<image_url>"
                            elif content["type"] == "text":
                                all_response += content["text"]
                        tool_messages[-1]["content"].extend(tool_message)
                    all_response += "</tool_response>"

            response = await client.chat.completions.create(
                model=self.model_version,
                messages=messages + tool_messages,
                max_tokens=gen_kwargs["max_new_tokens"],
                temperature=gen_kwargs["temperature"],
                tools=functions,
                tool_choice="auto",
                extra_body=copy.deepcopy(self.request_extra_body) if self.request_extra_body else None,
            )
            last_response = self._coerce_message_content(response.choices[0].message)
            all_response += last_response
        return all_response, idx

    def generate_until(self, requests) -> List[str]:
        async def run():
            pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")
            current_concurrency = self._get_initial_concurrency()
            dispatch_order = self._compute_dispatch_order(requests)
            res = await self._run_scheduling_loop(requests, dispatch_order, pbar, current_concurrency)
            pbar.close()
            return res

        eval_results = asyncio.run(run())
        eval_results.sort(key=lambda x: x[1])
        results = [content for content, _ in eval_results]
        if self.rank == 0 and len(self.base_urls) > 1:
            usage_summary = ", ".join(f"{url}={self._endpoint_request_counts.get(url, 0)}" for url in self.base_urls)
            eval_logger.info("AsyncOpenAI endpoint usage: {}", usage_summary)
        if self.mcp_client is not None:
            shutil.rmtree(self.work_dir)
        return results
