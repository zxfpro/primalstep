import os
from abc import ABC, abstractmethod
from openai import OpenAI

class BaseLLMClient(ABC):
    """
    定义所有LLM客户端必须遵循的抽象基类。
    """
    @abstractmethod
    def generate(self, prompt: str) -> str:
        """
        接收提示并返回LLM的原始字符串响应。
        """
        pass

class OpenAIClient(BaseLLMClient):
    """
    实现与OpenAI API的交互。
    """
    def __init__(self, api_key: str = None, model_name: str = "gpt-4o"):
        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OpenAI API Key未提供。请通过参数或环境变量OPENAI_API_KEY提供。")
        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name

    def generate(self, prompt: str) -> str:
        """
        调用OpenAI API，并返回响应内容。
        尝试使用 response_format={"type": "json_object"} 强制JSON输出。
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "你是一个任务分解助手，请严格按照用户要求输出JSON格式。"},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"}
            )
            return response.choices[0].message.content
        except Exception as e:
            raise RuntimeError(f"调用OpenAI API失败: {e}")