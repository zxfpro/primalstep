import argparse
import uvicorn
import os
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from primalstep.log import Log
from primalstep.core import TaskDecomposer
from primalstep.llm_integration.clients import BaseLLMClient, OpenAIClient
from primalstep.llm_integration.mock_clients import MockLLMClient

# 初始化日志
logger = Log.logger

# 定义请求和响应模型
class DecomposeRequest(BaseModel):
    goal: str

class DecomposeResponse(BaseModel):
    graph_nodes: List[Dict[str, Any]]
    graph_edges: List[List[str]]
    steps_details: Dict[str, Any]

app = FastAPI(
    title="PrimalStep API",
    description="API for intelligent task decomposition.",
    version="1.0.0",
)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有来源，生产环境应限制
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局变量，用于存储TaskDecomposer实例
task_decomposer: TaskDecomposer = None

@app.on_event("startup")
async def startup_event():
    global task_decomposer
    
    # 尝试从命令行参数解析环境和端口，或者使用默认值
    # 注意：当通过 uvicorn primalstep.server:app 启动时，argparse 不会执行
    # 但为了在直接运行 python server.py 时也能配置，这里保留
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8008)
    parser.add_argument('--env', type=str, default='dev', choices=['dev', 'prod'])
    
    # parse_known_args() 可以解析已知参数，忽略未知参数
    # 这对于 uvicorn 启动时，uvicorn 自己的参数不会导致 argparse 报错很有用
    args, unknown = parser.parse_known_args()

    env = args.env
    port = args.port

    if env == "dev":
        Log.reset_level('debug', env=env)
        logger.info("开发环境启动...")
        llm_client: BaseLLMClient = MockLLMClient()
    elif env == "prod":
        Log.reset_level('info', env=env)
        logger.info("生产环境启动...")
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.critical("生产环境缺少OPENAI_API_KEY环境变量！")
            raise ValueError("OPENAI_API_KEY环境变量未设置。")
        llm_client = OpenAIClient(api_key=api_key)
    
    task_decomposer = TaskDecomposer(llm_client=llm_client)
    logger.info("TaskDecomposer 初始化完成。")

@app.post("/decompose", response_model=DecomposeResponse)
async def decompose_task_endpoint(request: DecomposeRequest):
    logger.info(f"接收到分解请求，目标: {request.goal}")
    try:
        graph, steps_details = task_decomposer.decompose_task(request.goal)
        
        # 将NetworkX图转换为适合API响应的格式
        graph_nodes = [{"id": node, **graph.nodes[node]} for node in graph.nodes()]
        graph_edges = [[u, v] for u, v in graph.edges()]
        
        logger.info("任务分解成功，返回结果。")
        return DecomposeResponse(
            graph_nodes=graph_nodes,
            graph_edges=graph_edges,
            steps_details=steps_details
        )
    except ValueError as e:
        logger.warning(f"分解请求失败 (客户端错误): {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"分解请求失败 (服务器内部错误): {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="内部服务器错误，请稍后再试。")

if __name__ == "__main__":
    # 这是一个标准的 Python 入口点惯用法
    # 当脚本直接运行时 (__name__ == "__main__")，这里的代码会被执行
    # 当通过 python -m YourPackageName 执行 __main__.py 时，__name__ 也是 "__main__"
    
    # 重新解析命令行参数，因为 uvicorn.run 需要它们
    parser = argparse.ArgumentParser(
        description="Start the PrimalStep FastAPI server."
    )
    parser.add_argument(
        '--port',
        type=int,
        default=8008,
        help='Specify the port for the server [default: 8008]'
    )
    parser.add_argument(
        '--env',
        type=str,
        default='dev',
        choices=['dev', 'prod'],
        help='Set the environment (dev or prod) [default: dev]'
    )
    parser.add_argument(
        '--reload',
        action='store_true',
        help='Enable auto-reload for development [default: False for prod, True for dev]'
    )

    args = parser.parse_args()

    # 根据环境调整端口和热重载
    run_port = args.port
    run_reload = args.reload

    if args.env == "dev":
        # 开发环境默认端口加100，并启用热重载
        if not args.port: # 如果用户没有指定端口，则使用默认端口+100
            run_port = 8108
        if not args.reload: # 如果用户没有指定reload，则开发环境默认开启
            run_reload = True
    
    # 注意：startup_event 会在 uvicorn.run 之前被调用，所以日志和 LLM 客户端的初始化会在那里完成。
    # 这里主要是为了配置 uvicorn 本身的运行参数。
    logger.info(f"Uvicorn 启动中，端口: {run_port}, 环境: {args.env}, 热重载: {run_reload}")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=run_port,
        reload=run_reload
    )
