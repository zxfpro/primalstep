import pytest
import networkx as nx
from unittest.mock import Mock, patch
from primalstep.core import TaskDecomposer
from primalstep.llm_integration.clients import BaseLLMClient
from primalstep.llm_integration.mock_clients import MockLLMClient

class TestTaskDecomposer:
    @pytest.fixture
    def mock_llm_client(self):
        return MockLLMClient()

    @pytest.fixture
    def decomposer(self, mock_llm_client):
        return TaskDecomposer(llm_client=mock_llm_client)

    def test_decompose_task_simple(self, decomposer):
        goal = "分解一个简单的任务"
        graph, steps_details = decomposer.decompose_task(goal)

        assert isinstance(graph, nx.DiGraph)
        assert "step1" in graph.nodes
        assert "step2" in graph.nodes
        assert graph.has_edge("step1", "step2")
        assert steps_details["step1"]["description"] == "第一步"
        assert steps_details["step2"]["description"] == "第二步"

    def test_decompose_task_invalid_json(self, decomposer):
        decomposer.llm_client.mock_response = "无效的JSON"
        with pytest.raises(ValueError, match="LLM响应格式不符合预期，缺少'steps'键。"):
            decomposer.decompose_task("测试无效JSON")

    def test_decompose_task_missing_steps_key(self, decomposer):
        decomposer.llm_client.mock_response = {"data": "some_data"}
        with pytest.raises(ValueError, match="LLM响应格式不符合预期，缺少'steps'键"):
            decomposer.decompose_task("测试缺少steps键")

    def test_decompose_task_steps_not_list(self, decomposer):
        decomposer.llm_client.mock_response = {"steps": "不是列表"}
        with pytest.raises(ValueError, match="LLM响应中的'steps'不是列表"):
            decomposer.decompose_task("测试steps不是列表")

    def test_decompose_task_missing_id_or_description(self, decomposer):
        decomposer.llm_client.mock_response = {"steps": [{"id": "step1"}]}
        with pytest.raises(ValueError, match="步骤数据缺少'id'或'description'"):
            decomposer.decompose_task("测试缺少id或description")

    def test_decompose_task_cyclic_dependency(self, decomposer):
        decomposer.llm_client.mock_response = {
            "steps": [
                {"id": "stepA", "description": "A", "dependencies": ["stepB"]},
                {"id": "stepB", "description": "B", "dependencies": ["stepA"]}
            ]
        }
        with pytest.raises(ValueError, match="检测到循环依赖"):
            decomposer.decompose_task("测试循环依赖")

    def test_decompose_task_llm_client_error(self, decomposer):
        decomposer.llm_client.error_mode = True
        with pytest.raises(RuntimeError, match="任务分解失败"):
            decomposer.decompose_task("测试LLM客户端错误")

    def test_build_llm_prompt(self, decomposer):
        goal = "创建一个网站"
        prompt = decomposer._build_llm_prompt(goal)
        assert "创建一个网站" in prompt
        assert "严格遵循以下JSON格式" in prompt
        assert '"id": "string"' in prompt
        assert '"description": "string"' in prompt
        assert '"dependencies": ["string"]' in prompt
        assert '"instructions": ["string"]' in prompt