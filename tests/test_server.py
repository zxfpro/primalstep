import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, Mock
import networkx as nx
import json

# 导入要测试的 FastAPI 应用和相关模块
from primalstep.server import app, task_decomposer
from primalstep.core import TaskDecomposer
from primalstep.llm_integration.mock_clients import MockLLMClient
from primalstep.llm_integration.clients import OpenAIClient

# 创建测试客户端
# client = TestClient(app) # 暂时注释掉，将在 fixture 中创建

@pytest.fixture(scope="module")
def test_app():
    """
    在测试模块开始时启动 FastAPI 应用，并在结束时关闭。
    """
    with TestClient(app) as client:
        yield client

@patch('primalstep.server.task_decomposer.decompose_task')
def test_decompose_endpoint_success(mock_decompose_task, test_app):
    mock_decompose_task.return_value = (
        nx.DiGraph([("step1", "step2")]),
        {"step1": {"id": "step1", "description": "Test Step 1", "dependencies": []},
         "step2": {"id": "step2", "description": "Test Step 2", "dependencies": ["step1"]}}
    )
    response = test_app.post("/decompose", json={"goal": "Test Goal"})
    assert response.status_code == 200
    data = response.json()
    assert "graph_nodes" in data
    assert "graph_edges" in data
    assert "steps_details" in data
    assert data["steps_details"]["step1"]["description"] == "Test Step 1"
    mock_decompose_task.assert_called_once_with("Test Goal")

@patch('primalstep.server.task_decomposer.decompose_task')
def test_decompose_endpoint_value_error(mock_decompose_task, test_app):
    mock_decompose_task.side_effect = ValueError("Invalid input")
    response = test_app.post("/decompose", json={"goal": "Invalid Goal"})
    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid input"

@patch('primalstep.server.task_decomposer.decompose_task')
def test_decompose_endpoint_internal_server_error(mock_decompose_task, test_app):
    mock_decompose_task.side_effect = Exception("Internal Error")
    response = test_app.post("/decompose", json={"goal": "Error Goal"})
    assert response.status_code == 500
    assert response.json()["detail"] == "内部服务器错误，请稍后再试。"

@patch('primalstep.server.OpenAIClient')
@patch('os.getenv', return_value='test_api_key')
@pytest.mark.asyncio
async def test_startup_event_prod_env(mock_getenv, MockOpenAIClient):
    # 模拟生产环境启动
    with patch('primalstep.server.argparse.ArgumentParser') as MockArgumentParser:
        mock_args = Mock()
        mock_args.env = 'prod'
        mock_args.port = 8000
        MockArgumentParser.return_value.parse_known_args.return_value = (mock_args, [])
        
        # 确保 task_decomposer 在测试前是 None，以便 startup_event 重新初始化
        global task_decomposer
        task_decomposer = None

        # 调用 startup_event
        with patch('primalstep.server.Log.reset_level') as mock_reset_level:
            with patch('primalstep.server.TaskDecomposer') as MockTaskDecomposer:
                # 模拟 TaskDecomposer 实例
                mock_decomposer_instance = Mock(spec=TaskDecomposer)
                MockTaskDecomposer.return_value = mock_decomposer_instance

                await app.router.startup() # 触发 startup 事件

                mock_reset_level.assert_called_once_with('info', env='prod')
                mock_getenv.assert_called_once_with("OPENAI_API_KEY")
                MockOpenAIClient.assert_called_once_with(api_key='test_api_key')
                MockTaskDecomposer.assert_called_once()
                assert task_decomposer is mock_decomposer_instance

@patch('primalstep.server.OpenAIClient')
@patch('os.getenv', return_value=None)
@pytest.mark.asyncio
async def test_startup_event_prod_env_no_api_key(mock_getenv, MockOpenAIClient):
    with patch('primalstep.server.argparse.ArgumentParser') as MockArgumentParser:
        mock_args = Mock()
        mock_args.env = 'prod'
        mock_args.port = 8000
        MockArgumentParser.return_value.parse_known_args.return_value = (mock_args, [])
        
        global task_decomposer
        task_decomposer = None

        with patch('primalstep.server.Log.reset_level'):
            with pytest.raises(ValueError, match="OPENAI_API_KEY环境变量未设置。"):
                await app.router.startup()
            mock_getenv.assert_called_once_with("OPENAI_API_KEY")
            MockOpenAIClient.assert_not_called()

@pytest.mark.asyncio
async def test_startup_event_dev_env():
    with patch('primalstep.server.argparse.ArgumentParser') as MockArgumentParser:
        mock_args = Mock()
        mock_args.env = 'dev'
        mock_args.port = 8008
        MockArgumentParser.return_value.parse_known_args.return_value = (mock_args, [])
        
        global task_decomposer
        task_decomposer = None

        with patch('primalstep.server.Log.reset_level') as mock_reset_level:
            with patch('primalstep.server.MockLLMClient') as MockMockLLMClient:
                # 模拟 TaskDecomposer 实例
                mock_decomposer_instance = Mock(spec=TaskDecomposer)
                MockTaskDecomposer = Mock(return_value=mock_decomposer_instance) # 模拟 TaskDecomposer 类
                with patch('primalstep.server.TaskDecomposer', new=MockTaskDecomposer):
                    await app.router.startup()

                    mock_reset_level.assert_called_once_with('debug', env='dev')
                    MockMockLLMClient.assert_called_once()
                    MockTaskDecomposer.assert_called_once()
                    assert task_decomposer is mock_decomposer_instance