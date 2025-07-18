import pytest
import json
from click.testing import CliRunner
from unittest.mock import patch, Mock
import networkx as nx
from primalstep.cli import cli
from primalstep.core import TaskDecomposer
from primalstep.llm_integration.mock_clients import MockLLMClient
from primalstep.llm_integration.clients import OpenAIClient

class TestCLI:
    @pytest.fixture
    def runner(self):
        return CliRunner()

    @pytest.fixture
    def mock_decomposer(self):
        # 创建一个模拟的TaskDecomposer实例
        mock_decomposer_instance = Mock(spec=TaskDecomposer)
        # 模拟decompose_task的返回值
        mock_decomposer_instance.decompose_task.return_value = (
            nx.DiGraph(), # 模拟一个空的DiGraph
            {"step1": {"description": "Test Step", "dependencies": []}}
        )
        return mock_decomposer_instance

    @patch('primalstep.cli.TaskDecomposer')
    def test_decompose_command_text_output(self, MockTaskDecomposer, runner, mock_decomposer):
        MockTaskDecomposer.return_value = mock_decomposer
        
        # 模拟LLM客户端，确保它返回一个有效的JSON字符串
        mock_llm_client = Mock(spec=MockLLMClient)
        mock_llm_client.generate.return_value = '{"steps": [{"id": "step1", "description": "Test Step", "dependencies": []}]}'
        
        # 确保TaskDecomposer的构造函数接收到模拟的LLM客户端
        MockTaskDecomposer.return_value.llm_client = mock_llm_client
        MockTaskDecomposer.return_value.decompose_task.return_value = (
            nx.DiGraph([("step1", "step2")]),
            {"step1": {"id": "step1", "description": "Test Step", "dependencies": []},
             "step2": {"id": "step2", "description": "Test Step 2", "dependencies": ["step1"]}}
        )

        result = runner.invoke(cli, ['decompose', 'Test Goal', '--output', 'text', '--mock-llm'])
        assert result.exit_code == 0
        assert "目标: Test Goal" in result.output
        assert "ID: step1" in result.output
        assert "描述: Test Step" in result.output
        assert "分解步骤" in result.output

    @patch('primalstep.cli.TaskDecomposer')
    def test_decompose_command_json_output(self, MockTaskDecomposer, runner, mock_decomposer):
        MockTaskDecomposer.return_value = mock_decomposer
        
        mock_llm_client = Mock(spec=MockLLMClient)
        mock_llm_client.generate.return_value = '{"steps": [{"id": "step1", "description": "Test Step", "dependencies": []}]}'
        MockTaskDecomposer.return_value.llm_client = mock_llm_client

        result = runner.invoke(cli, ['decompose', 'Test Goal', '--output', 'json', '--mock-llm'])
        assert result.exit_code == 0
        output_json = json.loads(result.output)
        assert output_json["goal"] == "Test Goal"
        assert output_json["steps_details"]["step1"]["description"] == "Test Step"

    @patch('primalstep.cli.TaskDecomposer')
    def test_decompose_command_llm_error(self, MockTaskDecomposer, runner):
        MockTaskDecomposer.return_value.decompose_task.side_effect = RuntimeError("LLM API Error")
        result = runner.invoke(cli, ['decompose', 'Error Goal', '--mock-llm'])
        assert result.exit_code == 1
        assert "发生意外错误: LLM API Error" in result.output

    @patch('primalstep.cli.TaskDecomposer')
    def test_decompose_command_value_error(self, MockTaskDecomposer, runner):
        MockTaskDecomposer.return_value.decompose_task.side_effect = ValueError("Invalid input")
        result = runner.invoke(cli, ['decompose', 'Invalid Goal', '--mock-llm'])
        assert result.exit_code == 1
        assert "Error: 任务分解失败 (输入或逻辑错误): Invalid input" in result.output

    @patch('primalstep.cli.OpenAIClient')
    @patch('os.getenv', return_value='test_api_key')
    def test_decompose_command_openai_client(self, mock_getenv, MockOpenAIClient, runner):
        # 模拟OpenAIClient的实例
        mock_openai_client_instance = Mock(spec=OpenAIClient)
        mock_openai_client_instance.generate.return_value = '{"steps": [{"id": "step1", "description": "OpenAI Test", "dependencies": []}]}'
        MockOpenAIClient.return_value = mock_openai_client_instance

        # 模拟TaskDecomposer的实例
        mock_decomposer_instance = Mock(spec=TaskDecomposer)
        mock_decomposer_instance.decompose_task.return_value = (
            nx.DiGraph(),
            {"step1": {"description": "OpenAI Test", "dependencies": []}}
        )
        # 确保TaskDecomposer的构造函数接收到模拟的LLM客户端
        mock_decomposer_instance.llm_client = mock_openai_client_instance
        mock_decomposer_instance.decompose_task.return_value = (
            nx.DiGraph([("step1", "step2")]),
            {"step1": {"id": "step1", "description": "OpenAI Test", "dependencies": []},
             "step2": {"id": "step2", "description": "OpenAI Test 2", "dependencies": ["step1"]}}
        )

        with patch('primalstep.cli.TaskDecomposer', return_value=mock_decomposer_instance):
            result = runner.invoke(cli, ['decompose', 'OpenAI Goal', '--no-mock-llm'])
            assert result.exit_code == 0
            assert "OpenAI Test" in result.output
            MockOpenAIClient.assert_called_once_with(api_key='test_api_key')

    @patch('primalstep.cli.OpenAIClient')
    @patch('os.getenv', return_value=None)
    def test_decompose_command_openai_client_no_api_key(self, mock_getenv, MockOpenAIClient, runner):
        result = runner.invoke(cli, ['decompose', 'OpenAI Goal', '--no-mock-llm'])
        assert result.exit_code == 1
        assert "OpenAI API Key未提供" in result.output