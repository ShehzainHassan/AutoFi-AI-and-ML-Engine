
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Callable
from app.models import model_persistance

class ModelServingService:
    def __init__(self, max_workers: int = 4):
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.models: Dict[str, any] = {}
        self.loading_tasks: Dict[str, asyncio.Task] = {}
        self.model_lock = asyncio.Lock()

        self.model_registry: Dict[str, Callable[[], any]] = {
            "collaborative": model_persistance.load_collaborative_model,
            "vehicle_similarity": model_persistance.load_content_model,
            "user_similarity": model_persistance.load_user_content_model,
        }

    async def load_model(self, model_name: str):
        if model_name in self.models:
            return self.models[model_name]

        if model_name in self.loading_tasks:
            return None

        if model_name not in self.model_registry:
            raise ValueError(f"Unknown model: {model_name}")

        async with self.model_lock:
            if model_name not in self.loading_tasks:
                loop = asyncio.get_event_loop()
                task = loop.create_task(self._load_model_async(model_name))
                self.loading_tasks[model_name] = task

        return None

    async def _load_model_async(self, model_name: str):
        loop = asyncio.get_event_loop()
        loader = self.model_registry[model_name]
        model = await loop.run_in_executor(self.executor, loader)
        self.models[model_name] = model
        self.loading_tasks.pop(model_name, None)
