"""Эвристический бэйзлайн оценки качества изображений (ядро из QualityJPG без интерфейса, БД и LLM)."""
from .pipeline import analyze, load_image

__all__ = ["analyze", "load_image"]
