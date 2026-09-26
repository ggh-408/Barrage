"""Barrage 纯视觉强化学习环境与训练工具。"""

def __getattr__(name):
    if name == "BarrageVisionEnv":
        from .env import BarrageVisionEnv
        globals()[name] = BarrageVisionEnv
        return BarrageVisionEnv
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["BarrageVisionEnv"]
