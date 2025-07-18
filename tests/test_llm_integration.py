import pytest
import json
import os
from unittest.mock import patch, Mock
from primalstep.llm_integration.clients import BaseLLMClient, OpenAIClient
from primalstep.llm_integration.mock_clients import MockLLMClient

class TestBaseLLMClient:
    def test_abstract_method(self):
        with pytest.raises(TypeError):
            BaseLLMClient()

class TestOpenAIClient:
    @patch('openai.OpenAI')
    def test_init_with_api_key(self, mock_openai):
        with patch('primalstep.llm_integration.clients.OpenAI', new=mock_openai):
            client = OpenAIClient(api_key="test_key")
            mock_openai.assert_called_once_with(api_key="test_key")
            assert client.model_name == "gpt-4o"

    @patch('openai.OpenAI')
    def test_init_from_env(self, mock_openai):
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'env_key'}):
            with patch('primalstep.llm_integration.clients.OpenAI', new=mock_openai):
                client = OpenAIClient()
                mock_openai.assert_called_once_with(api_key="env_key")

    def test_init_no_api_key(self):
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="OpenAI API Key未提供"):
                OpenAIClient()

    @patch('primalstep.llm_integration.clients.OpenAIClient.generate')
    def test_generate_success(self, mock_generate):
        mock_generate.return_value = "LLM Response"
        client = OpenAIClient(api_key="test_key")
        response = client.generate("Test prompt")
        assert response == "LLM Response"
        mock_generate.assert_called_once_with("Test prompt")

    @patch('primalstep.llm_integration.clients.OpenAIClient.generate')
    def test_generate_api_error(self, mock_generate):
        mock_generate.side_effect = RuntimeError("调用OpenAI API失败: API Error")
        client = OpenAIClient(api_key="test_key")
        with pytest.raises(RuntimeError, match="调用OpenAI API失败: API Error"):
            client.generate("Test prompt")

class TestMockLLMClient:
    def test_init(self):
        client = MockLLMClient(mock_response={"test": "response"}, delay=0.5, error_mode=True)
        assert client.mock_response == {"test": "response"}
        assert client.delay == 0.5
        assert client.error_mode is True

    def test_generate_mock_response(self):
        mock_data = {"steps": [{"id": "mock", "description": "mocked"}]}
        client = MockLLMClient(mock_response=mock_data)
        response = client.generate("any prompt")
        assert response == json.dumps(mock_data)

    def test_generate_error_mode(self):
        client = MockLLMClient(error_mode=True)
        with pytest.raises(RuntimeError, match="Mock LLM 模拟错误"):
            client.generate("any prompt")

    def test_generate_default_simple_task(self):
        client = MockLLMClient()
        response = client.generate("分解一个简单的任务")
        data = json.loads(response)
        assert "step1" in [s["id"] for s in data["steps"]]
        assert "step2" in [s["id"] for s in data["steps"]]

    def test_generate_default_cyclic_dependency(self):
        client = MockLLMClient()
        response = client.generate("循环依赖")
        data = json.loads(response)
        assert "stepA" in [s["id"] for s in data["steps"]]
        assert "stepB" in [s["id"] for s in data["steps"]]
        assert data["steps"][0]["dependencies"] == ["stepB"]
        assert data["steps"][1]["dependencies"] == ["stepA"]

    def test_generate_default_other(self):
        client = MockLLMClient()
        response = client.generate("其他任务")
        data = json.loads(response)
        assert "default_step" in [s["id"] for s in data["steps"]]