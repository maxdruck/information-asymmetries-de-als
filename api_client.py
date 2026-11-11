"""
Provides the OpenAI-compatible AcademicCloud client with key rotation.

Expected format of the project's ``.env`` file:

  API_KEY_1=...
  API_KEY_2=...
  API_KEY_3=...

Optional:
  API_KEY_COOLDOWN_SECONDS=3600
  API_ALL_KEYS_WAIT_SECONDS=3600

Behavior:
  - Available keys are used in turn.
  - On a 429 error, the current key is put on cooldown.
  - On a 500 error, the next available key is tried.
  - If all keys are temporarily blocked, the client waits automatically.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

import openai
from dotenv import dotenv_values
from openai import InternalServerError, RateLimitError

from pipeline_config import BASE_URL, ENV_PATH


DEFAULT_KEY_COOLDOWN_SECONDS = 3600
DEFAULT_ALL_KEYS_WAIT_SECONDS = 3600
DEFAULT_TIMEOUT_SECONDS = 120.0


# Numeric API settings can come from environment variables or .env
def _env_int(name: str, default: int) -> int:
    config = dotenv_values(ENV_PATH)
    value = os.getenv(name) or config.get(name)

    if value in (None, ""):
        return default

    try:
        return int(value)
    except ValueError:
        return default


def _load_api_keys() -> list[tuple[str, str]]:
    """Load the API keys from the project's ``.env`` file.

    Preferred names:
      API_KEY_1
      API_KEY_2
      API_KEY_3

    Fallback:
      API_KEY
    """
    config = dotenv_values(ENV_PATH)

    keys: list[tuple[str, str]] = []
    seen_values: set[str] = set()

    # Duplicate keys are registered only once
    for name in ("API_KEY_1", "API_KEY_2", "API_KEY_3"):
        value = os.getenv(name) or config.get(name)

        if value and value not in seen_values:
            keys.append((name, value))
            seen_values.add(value)

    fallback = os.getenv("API_KEY") or config.get("API_KEY")

    if not keys and fallback:
        keys.append(("API_KEY", fallback))

    if not keys:
        raise ValueError(
            "No API key found. Add API_KEY_1, API_KEY_2, API_KEY_3 "
            "or API_KEY to .env."
        )

    return keys


def _seconds_from_rate_limit_error(error: RateLimitError) -> int | None:
    """Read the reset time from the rate-limit headers, if available.

    Supported header formats:
      ratelimit-reset: 3127
      x-ratelimit-reset: 3127

    If no usable header is present, ``None`` is returned.
    """
    response = getattr(error, "response", None)
    headers = getattr(response, "headers", None)

    if not headers:
        return None

    # AcademicCloud may provide the reset time under several common header names
    for header_name in (
        "ratelimit-reset",
        "x-ratelimit-reset",
        "x-ratelimit-reset-hour",
        "x-ratelimit-reset-minute",
    ):
        raw_value = headers.get(header_name)

        if raw_value is None:
            continue

        text = str(raw_value).strip()

        try:
            value = float(text)
        except ValueError:
            continue

        # Most common case: the value directly specifies a number of seconds
        if 0 < value <= 24 * 60 * 60:
            return int(value) + 5

        # Alternatively, the value may be a Unix timestamp
        now = time.time()
        if value > now:
            return int(value - now) + 5

    return None


@dataclass
class APIKeySlot:
    name: str
    key: str
    client: openai.OpenAI
    cooldown_until: float = 0.0


# This facade transparently distributes requests across all configured keys
class RotatingAcademicCloudClient:
    """Manages several OpenAI-compatible AcademicCloud clients.

    Main method:
      create_chat_completion(...)
    """

    def __init__(
        self,
        base_url: str = BASE_URL,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        key_cooldown_seconds: int = DEFAULT_KEY_COOLDOWN_SECONDS,
        all_keys_wait_seconds: int = DEFAULT_ALL_KEYS_WAIT_SECONDS,
    ) -> None:
        self.key_cooldown_seconds = key_cooldown_seconds
        self.all_keys_wait_seconds = all_keys_wait_seconds
        self.active_index = 0

        self.slots: list[APIKeySlot] = []

        for name, key in _load_api_keys():
            client = openai.OpenAI(
                api_key=key,
                base_url=base_url,
                timeout=timeout,
            )
            self.slots.append(APIKeySlot(name=name, key=key, client=client))

        print(
            "AcademicCloud API client initialized with "
            f"{len(self.slots)} key(s): "
            + ", ".join(slot.name for slot in self.slots)
        )

    def _available_indices_ordered(self) -> list[int]:
        """Yield available key indices starting from the currently active index."""
        now = time.time()
        n_slots = len(self.slots)
        indices: list[int] = []

        for offset in range(n_slots):
            index = (self.active_index + offset) % n_slots
            slot = self.slots[index]

            if slot.cooldown_until <= now:
                indices.append(index)

        return indices

    def _seconds_until_next_key(self) -> int:
        """Determine the waiting time until the next available key."""
        now = time.time()
        next_time = min(slot.cooldown_until for slot in self.slots)
        wait_seconds = max(1, int(next_time - now))

        return min(wait_seconds, self.all_keys_wait_seconds)

    def _wait_until_key_available(self) -> None:
        wait_seconds = self._seconds_until_next_key()
        wait_seconds = max(wait_seconds, 60)

        print(
            "All API keys are currently on cooldown. "
            f"Waiting {wait_seconds} seconds..."
        )

        time.sleep(wait_seconds)

    def create_chat_completion(self, **kwargs):
        """Create a chat completion.

        On a ``RateLimitError``, the current key is temporarily suspended and
        the next available one is used. If all keys are affected, the client
        waits. On an ``InternalServerError``, the same request is repeated
        with the next key; if all attempts fail, the last server error is
        re-raised.
        """
        # The loop only ends with a response or an unrecoverable error
        while True:
            available_indices = self._available_indices_ordered()

            if not available_indices:
                self._wait_until_key_available()
                continue

            last_server_error: InternalServerError | None = None

            for slot_index in available_indices:
                slot = self.slots[slot_index]
                self.active_index = (slot_index + 1) % len(self.slots)

                try:
                    return slot.client.chat.completions.create(**kwargs)

                except RateLimitError as error:
                    wait_seconds = (
                        _seconds_from_rate_limit_error(error)
                        or self.key_cooldown_seconds
                    )

                    slot.cooldown_until = time.time() + wait_seconds

                    print(
                        f"{slot.name} reached the rate limit. "
                        f"Cooling down for {wait_seconds} seconds."
                    )

                    continue

                except InternalServerError as error:
                    last_server_error = error

                    print(
                        f"{slot.name} returned a server error. "
                        "Trying the next API key..."
                    )

                    continue

            if last_server_error is not None:
                raise last_server_error

            # After pure rate-limit errors, the outer loop starts over
            # and waits there if all keys are cooling down by then
            continue


def load_rotating_kisski_client() -> RotatingAcademicCloudClient:
    # A central constructor keeps client creation identical across all callers
    return RotatingAcademicCloudClient(
        base_url=BASE_URL,
        timeout=DEFAULT_TIMEOUT_SECONDS,
        key_cooldown_seconds=_env_int(
            "API_KEY_COOLDOWN_SECONDS",
            DEFAULT_KEY_COOLDOWN_SECONDS,
        ),
        all_keys_wait_seconds=_env_int(
            "API_ALL_KEYS_WAIT_SECONDS",
            DEFAULT_ALL_KEYS_WAIT_SECONDS,
        ),
    )
