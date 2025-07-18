import json
import networkx as nx
from typing import List, Dict, Any, Tuple

from primalstep.llm_integration.clients import BaseLLMClient
from primalstep.utils.graph_helpers import validate_dag
from primalstep.log import Log

class TaskDecomposer:
    """
    `TaskDecomposer` 类封装了任务分解的核心逻辑。

    它负责协调大型语言模型（LLM）的调用、解析LLM返回的JSON数据，
    以及构建表示任务步骤和依赖关系的NetworkX有向无环图（DAG）。
    通过将LLM客户端作为依赖注入，实现了LLM实现的解耦。
    """
    def __init__(self, llm_client: BaseLLMClient):
        """
        初始化 `TaskDecomposer` 实例。

        Args:
            llm_client (BaseLLMClient): 一个实现了 `BaseLLMClient` 接口的LLM客户端实例。
                                         用于与LLM进行交互，生成任务分解结果。
        """
        self.llm_client = llm_client
        self.logger = Log.logger

    def decompose_task(self, goal_string: str) -> Tuple[nx.DiGraph, Dict[str, Any]]:
        """
        将高层级用户目标分解为一系列可执行的步骤和它们之间的依赖关系。

        该方法通过调用LLM生成任务分解的JSON表示，然后解析该JSON，
        并构建一个表示任务流程的NetworkX有向无环图（DAG）。

        Args:
            goal_string (str): 用户的高层级目标描述，例如“开发一个简单的待办事项应用”。

        Returns:
            Tuple[nx.DiGraph, Dict[str, Any]]: 一个包含两个元素的元组。
                - `nx.DiGraph`: 表示任务步骤及其依赖关系的NetworkX有向无环图。
                                图中的每个节点ID对应一个步骤ID。
                - `Dict[str, Any]`: 一个字典，键是步骤ID，值是包含步骤详细信息（如描述、指令、依赖）的字典。

        Raises:
            ValueError: 如果LLM返回的JSON格式无效、缺少关键数据（如'steps'键），
                        或者解析后的步骤数据不符合预期（如缺少'id'或'description'），
                        或者检测到循环依赖（图不是DAG）。
            RuntimeError: 如果在LLM调用或任务分解过程中发生任何其他意外错误。
        """
        self.logger.info(f"开始分解任务: {goal_string}")
        try:
            # 1. 调用 _build_llm_prompt() 生成LLM提示。
            prompt = self._build_llm_prompt(goal_string)
            self.logger.debug(f"LLM提示: {prompt}")

            # 2. 调用 self.llm_client.generate() 获取LLM响应（JSON字符串）。
            llm_response_str = self.llm_client.generate(prompt)
            self.logger.debug(f"LLM原始响应: {llm_response_str}")

            # 3. 解析LLM返回的JSON字符串为Python字典。
            try:
                llm_response_data = json.loads(llm_response_str)
            except json.JSONDecodeError as e:
                self.logger.error(f"LLM响应JSON解析失败: {e}")
                raise ValueError(f"LLM返回了无效的JSON格式: {e}")

            if not isinstance(llm_response_data, dict) or "steps" not in llm_response_data:
                self.logger.error("LLM响应缺少'steps'键或格式不正确。")
                raise ValueError("LLM响应格式不符合预期，缺少'steps'键。")

            steps_data = llm_response_data["steps"]
            if not isinstance(steps_data, list):
                self.logger.error("LLM响应中的'steps'不是列表。")
                raise ValueError("LLM响应中的'steps'不是列表。")

            graph = nx.DiGraph()
            steps_details = {}

            # 4. 遍历JSON中的步骤数据，构建NetworkX DiGraph：
            for step in steps_data:
                step_id = step.get("id")
                description = step.get("description")
                dependencies = step.get("dependencies", [])
                instructions = step.get("instructions", [])

                if not step_id or not description:
                    self.logger.error(f"步骤数据缺少'id'或'description': {step}")
                    raise ValueError(f"步骤数据缺少'id'或'description': {step}")

                # 每个步骤的 id 作为节点。
                # description 和 instructions 作为节点属性存储。
                graph.add_node(step_id, description=description, instructions=instructions)
                steps_details[step_id] = {
                    "description": description,
                    "instructions": instructions,
                    "dependencies": dependencies
                }

                # 根据 dependencies 字段添加有向边。
                for dep_id in dependencies:
                    if dep_id not in steps_details:
                        self.logger.warning(f"步骤 '{step_id}' 依赖于不存在的步骤 '{dep_id}'。")
                        # 可以在这里选择抛出错误或忽略，根据需求而定
                        # raise ValueError(f"步骤 '{step_id}' 依赖于不存在的步骤 '{dep_id}'。")
                    graph.add_edge(dep_id, step_id)

            # 5. 调用 primalstep.utils.graph_helpers.validate_dag() 验证图是否为DAG。
            validate_dag(graph) # 如果不是DAG，会抛出ValueError

            self.logger.info(f"任务分解成功，生成了包含 {graph.number_of_nodes()} 个节点和 {graph.number_of_edges()} 条边的DAG。")
            return graph, steps_details

        except ValueError as ve:
            self.logger.error(f"任务分解业务逻辑错误: {ve}")
            raise ve
        except Exception as e:
            self.logger.critical(f"任务分解过程中发生意外错误: {e}", exc_info=True)
            raise RuntimeError(f"任务分解失败: {e}")

    def _build_llm_prompt(self, goal: str) -> str:
        """
        根据用户提供的目标，构建一个详细的LLM提示字符串。

        该提示明确指示LLM将目标分解为JSON格式的步骤，并指定了JSON的结构、
        每个字段的含义、步骤的粒度以及依赖关系的要求（必须形成DAG）。

        Args:
            goal (str): 用户的高层级目标字符串。

        Returns:
            str: 格式化后的LLM提示字符串，可以直接发送给LLM客户端。
        """
        prompt = f"""
你是一个高级任务分解助手。你的任务是将一个高层级的用户目标分解成一系列清晰、可执行的原子步骤。
每个步骤都必须有一个唯一的ID，一个描述，以及一个可选的依赖步骤列表和可选的机器指令列表。
请确保所有步骤形成一个有向无环图（DAG），即没有循环依赖。

输出必须严格遵循以下JSON格式：
{{
  "steps": [
    {{
      "id": "string", // 唯一的步骤标识符，例如 "step1", "task_setup", "data_processing"
      "description": "string", // 对该步骤的简短描述，例如 "初始化项目", "收集用户输入"
      "dependencies": ["string"], // 可选，该步骤依赖的其他步骤的ID列表。如果无依赖，则为空列表。
      "instructions": ["string"] // 可选，该步骤的详细机器可执行指令列表，例如命令行命令、代码片段等。
    }}
  ]
}}

请注意以下几点：
1.  `id` 必须是字符串，且在所有步骤中唯一。
2.  `description` 必须是字符串，简洁明了。
3.  `dependencies` 必须是一个字符串列表，其中每个字符串都是其他步骤的 `id`。如果一个步骤没有依赖，`dependencies` 字段应为空列表 `[]`。
4.  `instructions` 必须是一个字符串列表，包含该步骤的具体执行指令。如果无指令，则为空列表 `[]`。
5.  确保所有 `dependencies` 中引用的 `id` 都存在于 `steps` 列表中。
6.  分解的粒度应适中，每个步骤应是相对独立的、可完成的单元。
7.  不要包含任何额外的文本或解释，只返回纯JSON。

用户目标: "{goal}"

请开始分解任务并生成JSON。
"""
        return prompt
