import os
from collections.abc import Generator
from typing import Any

from google import genai
from google.genai.types import GenerateContentConfig


class GoogleVertexLLM:
    """Google Vertex AI / Gemini LLM client."""

    DEFAULT_MODEL = "gemini-2.5-flash-lite"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        api_key: str | None = None,
        project: str | None = None,
        location: str | None = None,
        system_instruction: str | None = None,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
    ) -> None:
        client_kwargs: dict[str, Any] = {}
        if api_key:
            client_kwargs["api_key"] = api_key
        if project and location:
            client_kwargs["vertexai"] = True
            client_kwargs["project"] = project
            client_kwargs["location"] = location

        self.client = genai.Client(**client_kwargs)
        self.model = model

        self._default_config: dict[str, Any] = {}
        if system_instruction is not None:
            self._default_config["system_instruction"] = system_instruction
        if temperature is not None:
            self._default_config["temperature"] = temperature
        if max_output_tokens is not None:
            self._default_config["max_output_tokens"] = max_output_tokens
        if top_p is not None:
            self._default_config["top_p"] = top_p
        if top_k is not None:
            self._default_config["top_k"] = top_k

    def _build_config(self, overrides: dict[str, Any] | None = None) -> GenerateContentConfig:
        merged = {**self._default_config, **(overrides or {})}
        return GenerateContentConfig(**merged) if merged else GenerateContentConfig()

    # ── Core APIs ──

    def generate(self, prompt: str, **kwargs: Any) -> str:
        """Single-turn text generation. Returns the model's text response."""
        config = self._build_config(kwargs)
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=config,
        )
        return response.text or ""

    def chat(
        self,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> str:
        """Multi-turn chat.

        Args:
            messages: List of ``{"role": "user"|"model", "text": "..."}`` dicts.
            **kwargs: Override generation config params.

        Returns:
            The model's text response for the last turn.
        """
        config = self._build_config(kwargs)
        from google.genai.types import Content, Part

        contents = [Content(role=m["role"], parts=[Part(text=m["text"])]) for m in messages]
        response = self.client.models.generate_content(
            model=self.model,
            contents=contents,
            config=config,
        )
        return response.text or ""

    def stream(self, prompt: str, **kwargs: Any) -> Generator[str]:
        """Streaming text generation. Yields text chunks."""
        config = self._build_config(kwargs)
        for chunk in self.client.models.generate_content_stream(
            model=self.model,
            contents=prompt,
            config=config,
        ):
            if chunk.text:
                yield chunk.text

    # ── Convenience ──
    def __call__(self, prompt: str, **kwargs: Any) -> str:
        """Shorthand: ``llm("Hello")`` is equivalent to ``llm.generate("Hello")``."""
        return self.generate(prompt, **kwargs)

    def __repr__(self) -> str:
        return f"GoogleVertexLLM(model={self.model!r})"


if __name__ == "__main__":
    """
    Example:

    ```bash
    python -m src.my_tool.googlevertex
    ```
    """

    # --- Simple Example ---
    print("--- Simple Example ---")
    llm = GoogleVertexLLM()
    response = llm("Hello, how are you?")
    print(response)

    # --- Chat Example ---
    print("--- Chat Example ---")
    messages = [
        {"role": "user", "text": "Hello, how are you?"},
        {"role": "model", "text": "I'm doing great, thank you!"},
    ]
    response = llm.chat(messages)
    print(response)

    # --- Stream Example ---
    print("--- Stream Example ---")
    for chunk in llm.stream("Hello, how are you?"):
        print(chunk, end="", flush=True)
    print()

    # --- Gemini API Example ---
    print("--- Gemini API Example ---")
    gemini_client = GoogleVertexLLM(
        model="gemini-2.5-flash-lite",
        api_key=os.getenv("GEMINI_API_KEY"),
    )
    print(gemini_client("Hello, how are you?"))

    # --- Vertex API Example ---
    print("--- Vertex API Example ---")
    vertex_client = GoogleVertexLLM(
        api_key=os.getenv("VERTEX_API_KEY"),
        project=os.getenv("GCP_PROJECT_ID"),
        location="us-central1",
        model="gemini-2.5-flash-lite",
    )

    print(vertex_client("Hello, how are you?"))
