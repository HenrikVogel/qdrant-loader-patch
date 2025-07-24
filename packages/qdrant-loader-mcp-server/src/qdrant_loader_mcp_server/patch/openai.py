from openai import AsyncOpenAI
from qdrant_loader_mcp_server.config import OpenAIConfig

class PatchedAsyncOpenAI(AsyncOpenAI):

    model: str

    def __init__(self, *args, model: str = "", **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model
    
    @classmethod
    def from_config(cls, config: OpenAIConfig):
        return cls(
            api_key=config.api_key,
            base_url=config.base_url,
            model = config.model
        )