"""Architecture-native chat rendering used by both generation and SFT."""

from __future__ import annotations


def single_turn_messages(prompt: str, response: str | None = None, system_prompt: str = "") -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    if response is not None:
        messages.append({"role": "assistant", "content": response})
    return messages
