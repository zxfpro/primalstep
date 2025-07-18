# PrimalStep 简易教程

欢迎使用 PrimalStep！本教程将引导您了解如何使用 PrimalStep 库的核心功能来分解复杂任务。

## 1. 核心概念

PrimalStep 的核心思想是将一个高层级的目标分解为一系列相互依赖的、可执行的原子步骤。这些步骤形成一个有向无环图（DAG），确保任务可以按正确的顺序执行。

主要组件：
- **`TaskDecomposer`**: 负责与大型语言模型（LLM）交互，将您的目标分解为结构化的步骤。
- **`BaseLLMClient`**: 一个抽象基类，定义了LLM客户端的接口。您可以实现自己的客户端来集成不同的LLM服务。

## 2. 安装

在您的Python环境中安装 PrimalStep：

```bash
pip install primalstep
```

## 3. 使用 `TaskDecomposer`

要分解一个任务，您需要：
1. 实例化一个 `BaseLLMClient` 的实现。
2. 实例化 `TaskDecomposer`，并将LLM客户端传递给它。
3. 调用 `decompose_task` 方法，传入您的目标字符串。

以下是一个简单的示例：

```python
import networkx as nx
from primalstep.core import TaskDecomposer
from primalstep.llm_integration.clients import BaseLLMClient

# 假设您有一个自定义的LLM客户端实现
# 这里我们使用一个模拟客户端作为示例
class MockLLMClient(BaseLLMClient):
    def generate(self, prompt: str) -> str:
        # 模拟LLM的响应，实际应用中这里会调用真实的LLM API
        # 为了简化，这里返回一个预设的JSON字符串
        return """
{
  "steps": [
    {
      "id": "step1",
      "description": "定义项目需求",
      "dependencies": [],
      "instructions": ["与客户沟通", "编写需求文档"]
    },
    {
      "id": "step2",
      "description": "设计数据库结构",
      "dependencies": ["step1"],
      "instructions": ["创建ER图", "定义表结构"]
    },
    {
      "id": "step3",
      "description": "开发后端API",
      "dependencies": ["step2"],
      "instructions": ["选择Web框架", "实现CRUD接口"]
    },
    {
      "id": "step4",
      "description": "开发前端界面",
      "dependencies": ["step1"],
      "instructions": ["选择前端框架", "设计UI/UX", "实现页面"]
    },
    {
      "id": "step5",
      "description": "集成前后端",
      "dependencies": ["step3", "step4"],
      "instructions": ["联调API", "部署到测试环境"]
    },
    {
      "id": "step6",
      "description": "测试与部署",
      "dependencies": ["step5"],
      "instructions": ["编写测试用例", "执行测试", "部署到生产环境"]
    }
  ]
}
"""

# 1. 实例化LLM客户端
mock_client = MockLLMClient()

# 2. 实例化任务分解器
decomposer = TaskDecomposer(mock_client)

# 3. 定义您的目标
goal = "开发一个简单的Web应用"

try:
    # 4. 分解任务
    task_graph, step_details = decomposer.decompose_task(goal)

    print(f"成功分解任务 '{goal}'。")
    print(f"生成的任务图包含 {task_graph.number_of_nodes()} 个步骤，{task_graph.number_of_edges()} 条依赖。")

    print("\n任务步骤详情:")
    for step_id, details in step_details.items():
        print(f"  ID: {step_id}")
        print(f"    描述: {details['description']}")
        print(f"    依赖: {details['dependencies']}")
        print(f"    指令: {details['instructions']}")

    print("\n任务执行顺序（拓扑排序）:")
    # 拓扑排序可以给出可能的执行顺序
    for i, node in enumerate(nx.topological_sort(task_graph)):
        print(f"  {i+1}. {node} - {step_details[node]['description']}")

except ValueError as e:
    print(f"任务分解失败: {e}")
except RuntimeError as e:
    print(f"发生意外错误: {e}")

```

## 4. 自定义LLM客户端

如果您想使用不同的LLM服务（例如OpenAI GPT、Google Gemini、Hugging Face模型等），您只需要实现 `BaseLLMClient` 抽象基类中定义的 `generate` 方法。

```python
from abc import ABC, abstractmethod

class BaseLLMClient(ABC):
    """
    LLM客户端的抽象基类。
    所有具体的LLM客户端实现都必须继承此基类并实现 `generate` 方法。
    """
    @abstractmethod
    def generate(self, prompt: str) -> str:
        """
        根据给定的提示生成LLM响应。

        Args:
            prompt (str): 发送给LLM的提示字符串。

        Returns:
            str: LLM生成的响应字符串。
        """
        raise NotImplementedError

# 示例：一个简单的OpenAI客户端（需要安装openai库）
# from openai import OpenAI

# class OpenAIClient(BaseLLMClient):
#     def __init__(self, api_key: str):
#         self.client = OpenAI(api_key=api_key)

#     def generate(self, prompt: str) -> str:
#         response = self.client.chat.completions.create(
#             model="gpt-3.5-turbo",
#             messages=[
#                 {"role": "system", "content": "You are a helpful assistant."},
#                 {"role": "user", "content": prompt}
#             ],
#             response_format={"type": "json_object"} # 确保返回JSON
#         )
#         return response.choices[0].message.content

# 使用方法：
# openai_client = OpenAIClient(api_key="YOUR_OPENAI_API_KEY")
# decomposer = TaskDecomposer(openai_client)
# task_graph, step_details = decomposer.decompose_task("开发一个天气预报应用")
```

## 5. 进一步探索

- **错误处理**: `decompose_task` 方法会捕获并抛出 `ValueError` 和 `RuntimeError`，您应该在您的应用程序中妥善处理这些异常。
- **日志**: PrimalStep 使用 `primalstep.log.Log` 进行日志记录，您可以配置日志级别以获取更详细的调试信息。
- **图操作**: `decompose_task` 返回的是一个 `networkx.DiGraph` 对象，您可以利用 NetworkX 库的强大功能对任务图进行进一步的分析和操作。

希望本教程能帮助您快速上手 PrimalStep！