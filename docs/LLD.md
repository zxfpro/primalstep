## 低层设计文档 (LLD) - 版本 1.0

### 1. 架构概述

本系统采用分层架构，核心业务逻辑位于 `primalstep/core.py`，外部交互（CLI、API）通过薄薄的适配层调用核心逻辑。LLM集成作为可插拔组件，通过抽象接口实现。日志系统作为共享基础设施，贯穿整个应用。

```
+----------------+       +-------------------+       +--------------------+
|  CLI (cli.py)  |------>|                   |       |                    |
+----------------+       |                   |<------|  LLM Provider A    |
                         |                   |       | (e.g., OpenAI API) |
+----------------+       |                   |       +--------------------+
| FastAPI Server |------>| TaskDecomposer    |
| (primalstep/   |       |  (primalstep/     |       +--------------------+
|  server.py)    |       |  core.py)         |       |  LLM Provider B    |
+----------------+       |                   |       | (e.g., Mock LLM)   |
                         |                   |       +--------------------+
+----------------+       |                   |
| Internal Tools |------>|                   |
| (e.g., tests)  |       +-------------------+
+----------------+              ^
                                |
                      +-------------------+
                      |   Logging System  |
                      | (primalstep/log.py)|
                      +-------------------+
```

### 2. 模块设计

#### 2.1 包结构 (基于提供的仓库结构)

```
primalstep/
├── README.md
├── dist/                          # 打包分发文件
│   └── primalstep-0.1.1-py3-none-any.whl
│   └── primalstep-0.1.1.tar.gz
├── docs/                          # 项目文档 (MkDocs)
│   └── index.md
├── main.py                        # (待定：考虑其作用或移除)
├── mkdocs.yml                     # MkDocs 配置文件
├── pyproject.toml                 # 项目元数据和依赖
├── src/
│   └── primalstep/                # 实际的 Python 包目录
│       ├── __init__.py            # 包初始化
│       ├── core.py                # 核心业务逻辑，所有公共API
│       ├── log.py                 # 日志系统实现
│       ├── server.py              # FastAPI 服务实现及入口点
│       └── llm_integration/       # LLM集成模块 (新增目录)
│           ├── __init__.py
│           ├── clients.py         # 抽象基类 BaseLLMClient 及真实LLM客户端实现
│           └── mock_clients.py    # Mock LLM客户端实现
│       └── utils/                 # 通用工具模块 (新增目录)
│           ├── __init__.py
│           └── graph_helpers.py   # NetworkX图相关辅助函数
│   └── primalstep.egg-info/       # (构建过程生成)
│       ├── PKG-INFO
│       ├── SOURCES.txt
│       ├── dependency_links.txt
│       ├── requires.txt
│       └── top_level.txt
├── tests/
│   └── test_main.py               # (需要调整为 test_core.py, test_server.py, test_cli.py 等)
└── uv.lock                        # (uv tool 的锁文件)
```

**说明：**
*   核心代码位于 `src/primalstep` 下。所有内部导入都应以 `from primalstep.<module> import ...` 形式进行。
*   `main.py` 的作用：当前 `server.py` 已包含完整的启动逻辑，`main.py` 可能不再需要。如果保留，其职责需明确，例如作为更高层次的应用入口或调度器。
*   `tests` 目录下的测试文件 `test_main.py` 需要重构，以符合模块化测试的最佳实践，例如拆分为 `test_core.py`, `test_server.py`, `test_cli.py` 等。

#### 2.2 `primalstep/core.py`

*   **`TaskDecomposer` 类：**
    *   **职责：** 封装任务分解的核心逻辑，协调LLM调用、JSON解析和NetworkX图构建。
    *   **`__init__(self, llm_client: BaseLLMClient)`：**
        *   接收一个实现了 `BaseLLMClient` 接口的实例。通过依赖注入实现LLM解耦。
    *   **`decompose_task(self, goal_string: str) -> tuple[nx.DiGraph, dict]`：**
        *   **输入：** `goal_string` (str) - 用户的高层级目标。
        *   **内部流程：**
            1.  调用 `_build_llm_prompt()` 生成LLM提示。
            2.  调用 `self.llm_client.generate()` 获取LLM响应（JSON字符串）。
            3.  解析LLM返回的JSON字符串为Python字典。
            4.  遍历JSON中的步骤数据，构建NetworkX `DiGraph`：
                *   每个步骤的 `id` 作为节点。
                *   `description` 和 `instructions` 作为节点属性存储。
                *   根据 `dependencies` 字段添加有向边。
            5.  调用 `primalstep.utils.graph_helpers.validate_dag()` 验证图是否为DAG。
        *   **输出：** `(nx.DiGraph, dict)` - NetworkX图和包含步骤详情的字典。
        *   **异常处理：** `ValueError` (JSON解析失败、无效数据、循环依赖)、`Exception` (LLM调用失败等)。
    *   **`_build_llm_prompt(self, goal: str) -> str` (私有方法)：**
        *   根据用户目标构建详细的LLM提示，明确要求输出JSON格式，并指定字段、粒度、依赖关系等。

#### 2.3 `primalstep/llm_integration/` (新增模块)

*   **`clients.py`：**
    *   **`BaseLLMClient` (抽象基类 `ABC`)：**
        *   **职责：** 定义所有LLM客户端必须遵循的接口。
        *   **`@abstractmethod generate(self, prompt: str) -> str`：** 抽象方法，接收提示并返回LLM的原始字符串响应。
    *   **`OpenAIClient(BaseLLMClient)`：**
        *   **职责：** 实现与OpenAI API的交互。
        *   **`__init__(self, api_key: str, model_name: str = "gpt-4o")`：** 初始化OpenAI客户端。
        *   **`generate(self, prompt: str) -> str`：** 调用OpenAI API，并返回响应内容。尝试使用 `response_format={"type": "json_object"}` 强制JSON输出。
*   **`mock_clients.py`：**
    *   **`MockLLMClient(BaseLLMClient)`：**
        *   **职责：** 用于开发和测试的模拟LLM客户端。
        *   **`__init__(self, mock_response: dict = None, delay: float = 0.1, error_mode: bool = False)`：**
            *   `mock_response`：预设的JSON响应数据。
            *   `delay`：模拟网络延迟。
            *   `error_mode`：模拟LLM调用失败。
        *   **`generate(self, prompt: str) -> str`：** 模拟LLM响应，根据 `prompt` 内容返回不同的预设JSON字符串。

#### 2.4 `primalstep/utils/` (新增模块)

*   **`graph_helpers.py`：**
    *   **`validate_dag(graph: nx.DiGraph) -> bool`：**
        *   **职责：** 验证给定的NetworkX图是否为有向无环图 (DAG)。
        *   **内部流程：** 调用 `nx.is_directed_acyclic_graph()`。
        *   **异常处理：** 如果检测到循环，抛出 `ValueError`。

#### 2.5 `primalstep/log.py` (现有实现)

*   **`Logger` 类 (单例模式)：**
    *   **职责：** 提供统一的日志记录接口，封装Python `logging` 模块。
    *   **`_instance`：** 用于实现单例的私有变量。
    *   **`__new__(cls, *args, **kwargs)`：** 确保只创建一个实例。
    *   **`__init__(self, level='debug', log_file_name="app.log")`：**
        *   初始化日志级别、日志文件路径（`logs/app.log`），并设置 `logging` 模块。
        *   支持 `RotatingFileHandler` (按大小轮转，最大 10MB，保留 5 个备份)。
    *   **`reset_level(self, level='debug', env='dev')`：**
        *   动态重置日志级别和环境。
        *   **实现细节：** 内部调用 `setup_logging`。为了避免重复添加 handlers，`setup_logging` 必须在重新配置时清除现有的 handlers。
    *   **`setup_logging(self)`：** 配置 `logging.Logger`、`Formatter` 和 `Handler`。
        *   **实现细节：**
            ```python
            # 在 setup_logging 内部，确保清除现有 handlers
            logger = logging.getLogger()
            # ... 其他配置 ...
            for handler in logger.handlers[:]: # 清除所有现有 handlers
                logger.removeHandler(handler)
            # ... 添加新的 handlers ...
            ```
*   **`Log = Logger(log_file_name = "app.log")`：** 在模块级别初始化 `Log` 单例，供其他模块导入使用。

#### 2.6 `primalstep/server.py` (现有实现与 FastAPI 集成)

*   **职责：** 作为 FastAPI 应用的入口点，负责启动HTTP服务。
*   **代码结构：**
    *   **FastAPI 应用实例 `app`：** 定义在文件顶部，包含 CORS 中间件配置。
    *   **`@app.on_event("startup")` 事件处理器：**
        *   在此处初始化全局的 `task_decomposer` 实例和日志 `logger`。
        *   读取命令行参数（通过 `argparse.parse_known_args()` 尝试解析 `port` 和 `env`），用于配置日志级别和启动信息。
        *   根据 `env` 参数（`dev`/`prod`）调用 `Log.reset_level()` 配置日志。
        *   实例化 `MockLLMClient` (默认) 或 `OpenAIClient` (生产环境根据环境变量配置)。
        *   实例化 `TaskDecomposer`。
    *   **`/decompose` API 端点：**
        *   **方法：** `POST`
        *   **请求模型：** `DecomposeRequest` (Pydantic 模型，包含 `goal: str`)。
        *   **响应模型：** `DecomposeResponse` (Pydantic 模型，包含 `graph_nodes`, `graph_edges`, `steps_details`)。
        *   **逻辑：** 调用全局 `task_decomposer.decompose_task()`，将结果转换为响应模型，并使用 `logger` 记录请求和错误。
        *   **错误处理：** 捕获 `ValueError` (返回 HTTP 400)，捕获其他 `Exception` (返回 HTTP 500)。
    *   **`if __name__ == "__main__":` 块：**
        *   **职责：** 允许直接运行 `python primalstep/server.py` 启动服务。
        *   **命令行解析：** 使用 `argparse` 解析 `port` 和 `env` 参数。
        *   **Uvicorn 启动：** 调用 `uvicorn.run()` 启动 `app`。
        *   **环境配置：** 根据 `env` 参数调整 `port` (开发环境 +100) 和 `reload` 状态 (开发环境 `True`，生产环境 `False`)。
        *   **注意：** `uvicorn` 通常通过 `uvicorn primalstep.server:app` 命令行启动，此时 `if __name__ == "__main__":` 块不会执行，`startup_event` 中的日志和分解器初始化仍会正确进行。

#### 2.7 `cli.py` (新增模块)

*   **职责：** 提供命令行界面，解析命令行参数，调用 `TaskDecomposer` 核心功能，并格式化输出。
*   **技术选型：** Click 库。
*   **命令：** `primalstep decompose`
    *   **参数：** `goal` (str, 必需)
    *   **选项：**
        *   `--output` (`-o`, `json` | `text`, 默认 `text`)：输出格式。
        *   `--mock-llm` (flag, 默认 `True`)：是否使用Mock LLM。
        *   `--api-key` (str, 可选)：真实LLM的API Key，从环境变量 `OPENAI_API_KEY` 读取。
*   **流程：**
    1.  根据 `--mock-llm` 选项实例化 `MockLLMClient` 或 `OpenAIClient`。
    2.  实例化 `primalstep.core.TaskDecomposer`。
    3.  调用 `decomposer.decompose_task()`。
    4.  根据 `--output` 选项将结果格式化为JSON或人类可读文本，并打印到标准输出。
    5.  使用 `primalstep.log.Log.logger` 记录CLI操作日志。
    6.  捕获并打印 `ValueError` 和其他异常。

#### 2.8 `pyproject.toml`

*   **职责：** 管理项目元数据、依赖、构建系统配置。
*   **重要性：** 它是现代Python项目打包和分发的标准，取代了 `setup.py` 和 `requirements.txt` 的部分功能。
*   **配置内容 (示例)：**
    *   `[project]`：
        *   `name = "primalstep"`
        *   `version = "0.1.0"` (或当前版本)
        *   `description = "Intelligent task decomposition tool"`
        *   `authors = [...]`
        *   `requires-python = ">=3.9"`
        *   `dependencies = ["fastapi", "uvicorn", "click", "networkx", "pydantic", "openai", ...]` (列出所有运行时依赖)
    *   `[build-system]`：指定构建后端。
    *   `[tool.poetry.scripts]` 或 `[project.scripts]`：定义CLI入口点，例如 `primalstep = "primalstep.cli:cli"`。

### 3. 数据结构

*   **LLM JSON 输出格式 (由LLM生成)：**
    ```json
    {
      "steps": [
        {
          "id": "string",
          "description": "string",
          "dependencies": ["string"], // 依赖的步骤ID列表
          "instructions": ["string"] // 可选的机器指令
        }
      ]
    }
    ```
*   **NetworkX 图：**
    *   **节点：** `id` (str)。
    *   **节点属性：** `description` (str), `instructions` (list[str])。
    *   **边：** `(source_step_id, target_step_id)`，表示 `source` 是 `target` 的前置依赖。

### 4. 错误处理

*   **LLM响应错误：**
    *   `json.JSONDecodeError`：LLM未返回有效JSON。`TaskDecomposer` 捕获并转换为 `ValueError`。
    *   LLM生成内容不符合预期结构（例如缺少 `id` 字段或依赖了不存在的步骤）：`TaskDecomposer` 内部校验并抛出 `ValueError`。
*   **循环依赖：**
    *   `primalstep.utils.graph_helpers.validate_dag()` 会检测到循环并抛出 `ValueError`。
*   **LLM API调用失败：**
    *   `BaseLLMClient` 的具体实现（如 `OpenAIClient`）会捕获API调用异常并转换为 `RuntimeError` 或其他特定异常。
*   **CLI/FastAPI 错误呈现：**
    *   将核心逻辑抛出的 `ValueError` 封装为用户友好的错误信息（CLI直接打印，FastAPI返回HTTP 400）。
    *   其他未捕获的异常统一作为内部错误处理（CLI打印通用错误，FastAPI返回HTTP 500）。
    *   所有错误都将通过 `primalstep.log.Log.logger` 进行记录，包括 WARN、ERROR 级别。

### 5. 测试策略

*   **单元测试 (`tests/test_core.py`, `tests/test_llm_integration.py`, `tests/test_utils.py`, `tests/test_log.py`)：**
    *   独立测试 `TaskDecomposer` 的每个方法。
    *   测试 `BaseLLMClient` 和其具体实现（包括Mock）。
    *   测试 `graph_helpers` 中的辅助函数。
    *   测试 `Log` 类的单例行为和日志配置功能。
    *   利用 `MockLLMClient` 模拟LLM的各种响应（成功、失败、无效JSON、循环依赖等），全面测试 `TaskDecomposer` 的健壮性。
*   **集成测试 (`tests/test_cli.py`, `tests/test_server.py`)：**
    *   测试CLI命令的端到端功能，包括参数解析、核心逻辑调用和输出格式。
    *   测试FastAPI服务的API端点，确保请求-响应流程正确。
    *   这些集成测试将主要依赖 `MockLLMClient`。

### 6. 部署考虑

*   **FastAPI：** 可通过 `uvicorn` 部署，并结合Gunicorn等WSGI服务器进行生产级部署。
*   **LLM API Key：** 建议通过环境变量（如 `OPENAI_API_KEY`）或秘密管理服务注入，不应硬编码。
*   **日志目录：** 确保部署环境有写入 `logs/` 目录的权限。
*   **Python 包：** `primalstep` 包可以通过 `pip install .` 或 `pip install -e .` (开发模式) 进行安装，使其模块可被Python解释器正确导入。
