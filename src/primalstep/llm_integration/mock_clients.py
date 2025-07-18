import json
import time
from .clients import BaseLLMClient

class MockLLMClient(BaseLLMClient):
    """
    用于开发和测试的模拟LLM客户端。
    """
    def __init__(self, mock_response: dict = None, delay: float = 0.1, error_mode: bool = False):
        self.mock_response = mock_response
        self.delay = delay
        self.error_mode = error_mode

    def generate(self, prompt: str) -> str:
        """
        模拟LLM响应，根据 prompt 内容返回不同的预设JSON字符串。
        """
        time.sleep(self.delay)
        if self.error_mode:
            raise RuntimeError("Mock LLM 模拟错误。")

        if self.mock_response:
            return json.dumps(self.mock_response)

        # 简单的模拟逻辑，可以根据prompt内容返回不同的响应
        if "分解一个简单的任务" in prompt:
            return json.dumps({
                "steps": [
                    {"id": "step1", "description": "第一步", "dependencies": []},
                    {"id": "step2", "description": "第二步", "dependencies": ["step1"]}
                ]
            })
        elif "循环依赖" in prompt:
            return json.dumps({
                "steps": [
                    {"id": "stepA", "description": "步骤A", "dependencies": ["stepB"]},
                    {"id": "stepB", "description": "步骤B", "dependencies": ["stepA"]}
                ]
            })
        else:
            return json.dumps({
                "steps": [
                    {"id": "default_step", "description": "默认模拟响应", "dependencies": []}
                ]
            })