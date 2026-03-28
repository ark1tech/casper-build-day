# Agent Customization Guide

Everything you need to edit is in `agent/src/agent/prompt.py`.

## The Two Things You Control

### 1. `SYSTEM_PROMPT`

This is the system prompt sent to your vision LLM with every frame. Tips:

- Be specific about the output format ("respond with 1-5 words")
- Tell the model to say "SKIP" if uncertain (saves guesses)
- Consider adding reasoning: "First describe what you see, then guess"
- Experiment with chain-of-thought vs. direct answers

### 2. `analyze(frame) -> str | None`

This function receives a `Frame` with:
- `frame.image` — a `PIL.Image.Image` (RGB)
- `frame.timestamp` — a `datetime` (UTC)

Return a guess string, or `None` to skip this frame.

## Example Implementation

```python
from pydantic_ai import Agent

from core import Frame
from agent.prompt import SYSTEM_PROMPT

agent = Agent("claude-sonnet-4-20250514", system_prompt=SYSTEM_PROMPT)

async def analyze(frame: Frame) -> str | None:
    result = await agent.run(
        "What is being shown? Give your best guess.",
        # Attach frame.image to your LLM call here
    )
    answer = result.output.strip()
    return None if answer == "SKIP" else answer
```

## Strategies to Try

- **Accumulate context**: Keep a history of recent frames to spot patterns
- **Confidence threshold**: Only guess when the model is highly confident
- **Multi-model**: Use a fast model for initial screening, a strong model for final guesses
- **Prompt iteration**: Test different prompts in practice mode before going live

## Practice Mode Tips

1. Point your camera at various objects
2. Run `uv run -m agent --practice`
3. Watch what your agent outputs
4. Tweak `SYSTEM_PROMPT` and `analyze()` until it reliably identifies things
5. Try `--fps 2` to see if more frames help your strategy
