import os
import asyncio
import time
import logging
from typing import List, Dict, Any, Optional
from openai import AsyncOpenAI, RateLimitError
from dotenv import load_dotenv

load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("LLMPool")

class LLMPool:
    def __init__(self):
        self.clients = {
            "nvidia": AsyncOpenAI(
                base_url="https://integrate.api.nvidia.com/v1",
                api_key=os.getenv("NVIDIA_API_KEY")
            ),
            "openrouter": AsyncOpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=os.getenv("OPENROUTER_API_KEY")
            ),
            "groq": AsyncOpenAI(
                base_url="https://api.groq.com/openai/v1",
                api_key=os.getenv("GROQ_API_KEY")
            ),
            "cerebras": AsyncOpenAI(
                base_url="https://api.cerebras.ai/v1",
                api_key="csk-pjcd94fk5rpxhmk52492cfekvddxtd89hmdky6fp5y9kdmnf"
            )
        }
        
        # Models ordered by quality/priority
        self.pool = [
            # ── 1. High Speed Tier (Cerebras) ────────────────────────
            {"id": "llama3.1-8b", "provider": "cerebras", "name": "Cerebras Llama 3.1 8B"},
            
            # ── 2. Verified High-RPM Tier (Groq) ─────────────────────
            {"id": "llama-3.3-70b-versatile", "provider": "groq", "name": "Groq Llama 3.3 70B"},
            
            # ── 3. Diverse Free Tier (OpenRouter) ───────────────────
            {"id": "google/gemma-4-26b-a4b-it:free", "provider": "openrouter", "name": "OR Gemma 4 26B"},
            {"id": "nvidia/nemotron-3-super-120b-a12b:free", "provider": "openrouter", "name": "OR Nemotron 120B"},
            {"id": "nousresearch/hermes-3-llama-3.1-405b:free", "provider": "openrouter", "name": "OR Llama 405B"},
            {"id": "qwen/qwen3-next-80b-a3b-instruct:free", "provider": "openrouter", "name": "OR Qwen 3 80B"},
        ]
        
        # Track current index for fallback rotation
        self.current_index = 0
        
        # Blacklist: {model_id: expiry_timestamp}
        self.blacklist: Dict[str, float] = {}
        self.blacklist_duration = 60  # seconds

        # Track active requests per model for true load balancing
        self.active_reqs: Dict[str, int] = {m["id"]: 0 for m in self.pool}

    def _is_blacklisted(self, model_id: str) -> bool:
        if model_id not in self.blacklist:
            return False
        if time.time() > self.blacklist[model_id]:
            del self.blacklist[model_id]
            return False
        return True

    async def get_completion(self, messages: List[Dict[str, str]], stream: bool = False, **kwargs) -> Any:
        """
        Attempts to get a completion using Smart Load Balancing.
        Picks the model with the lowest number of active requests first.
        """
        last_error = None
        
        # Sort pool by (active_reqs, priority)
        # Priority is implied by original pool order
        sorted_pool = sorted(
            self.pool, 
            key=lambda m: (self.active_reqs.get(m["id"], 0), self._is_blacklisted(m["id"]))
        )

        for model_cfg in sorted_pool:
            model_id = model_cfg["id"]
            provider = model_cfg["provider"]
            
            if self._is_blacklisted(model_id):
                continue
            
            client = self.clients.get(provider)
            if not client:
                continue

            # Increment load counter
            self.active_reqs[model_id] += 1
            
            # Fix for empty system prompts (some providers crash on None or empty string)
            sanitized_messages = []
            for m in messages:
                content = m.get("content")
                if not content or content.strip() == "":
                    content = "Keep this context in mind."
                sanitized_messages.append({"role": m["role"], "content": content})

            try:
                logger.info(f"[POOL] Using {model_cfg['name']} (active_reqs={self.active_reqs[model_id]}, stream={stream})...")
                
                response = await client.chat.completions.create(
                    model=model_id,
                    messages=sanitized_messages,
                    stream=stream,
                    **kwargs
                )
                
                if stream:
                    self.active_reqs[model_id] -= 1
                    return response 
                
                if not response or not hasattr(response, "choices") or not response.choices:
                    raise Exception("Empty response from API")

                content = response.choices[0].message.content
                if content:
                    self.active_reqs[model_id] -= 1
                    return content
                
            except RateLimitError as e:
                self.active_reqs[model_id] -= 1
                logger.warning(f"[POOL] 429 Rate Limit hit for {model_id}. Blacklisting for {self.blacklist_duration}s.")
                self.blacklist[model_id] = time.time() + self.blacklist_duration
                last_error = e
                continue 
                
            except Exception as e:
                self.active_reqs[model_id] -= 1
                logger.error(f"[POOL] Unexpected error with {model_id}: {str(e)}")
                last_error = e
                continue 

        # If all models failed
        error_msg = f"All models in pool failed. Last error: {str(last_error)}"
        logger.error(f"[POOL] FATAL: {error_msg}")
        raise last_error or Exception(error_msg)

        # If all models failed
        error_msg = f"All models in pool failed. Last error: {str(last_error)}"
        logger.error(f"[POOL] FATAL: {error_msg}")
        raise last_error or Exception(error_msg)

# Singleton instance
global_pool = LLMPool()

async def pool_complete(prompt: str, system_prompt: str = "You are a helpful assistant.", stream: bool = False, **kwargs) -> Any:
    """Wrapper for LightRAG and LongWriter compatibility."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]
    # Remove unsupported kwargs if any (LightRAG sends some)
    supported_args = ["temperature", "top_p", "max_tokens"]
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in supported_args}
    
    return await global_pool.get_completion(messages, stream=stream, **filtered_kwargs)
