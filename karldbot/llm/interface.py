from typing import Any, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class LLMInterface:
    def __init__(self, model: str, provider: str = "openai", retries: int = 3):
        self.model = model
        self.provider = provider
        self.retries = retries
        self._structured_llm = None
        self._lang_model = None
        self._initialize_models()

    def _initialize_models(self):
        try:
            from base_agent.llminterface import LangModel, StructuredLangModel

            self._structured_llm = StructuredLangModel(
                self.model, provider=self.provider, retries=self.retries
            )
            self._lang_model = LangModel(self.model, provider=self.provider)
        except ImportError:
            self._structured_llm = None
            self._lang_model = None

    def get_response(self, prompt: str, context: str = "") -> str:
        if self._lang_model is None:
            raise RuntimeError("base_agent not available")
        return self._lang_model.get_response(prompt, context)

    def get_structured_response(
        self, prompt: str, response_model: type[T], context: str = ""
    ) -> T:
        if self._structured_llm is None:
            raise RuntimeError("base_agent not available")
        return self._structured_llm.get_response(prompt, context, response_model)


class AsyncLLMInterface:
    def __init__(self, model: str, provider: str = "openai", retries: int = 3):
        self.model = model
        self.provider = provider
        self.retries = retries
        self._sync_interface = LLMInterface(model, provider, retries)

    async def get_response(self, prompt: str, context: str = "") -> str:
        import asyncio

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self._sync_interface.get_response, prompt, context
        )

    async def get_structured_response(
        self, prompt: str, response_model: type[T], context: str = ""
    ) -> T:
        import asyncio

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self._sync_interface.get_structured_response,
            prompt,
            response_model,
            context,
        )
