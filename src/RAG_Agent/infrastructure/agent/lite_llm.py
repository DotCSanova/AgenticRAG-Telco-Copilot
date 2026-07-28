from __future__ import annotations

from google.adk.models.lite_llm import LiteLlm as _LiteLlm


class LiteLlm(_LiteLlm):
    """LiteLlm with safe ``model_dump`` (workaround for google/adk-python#5367).

    Remove once upstream serializes ``LiteLLMClient`` correctly.
    """

    def model_dump(self, *args, **kwargs):  # noqa: ANN002, ANN003
        return {"model": self.model, "type": "LiteLlm"}
