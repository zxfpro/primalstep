import click
import json
import os
import networkx as nx

from primalstep.log import Log
from primalstep.core import TaskDecomposer
from primalstep.llm_integration.clients import BaseLLMClient, OpenAIClient
from primalstep.llm_integration.mock_clients import MockLLMClient

logger = Log.logger

@click.group()
def cli():
    """PrimalStep CLI for task decomposition."""
    pass

@cli.command()
@click.argument('goal', type=str)
@click.option('--output', '-o', type=click.Choice(['json', 'text']), default='text',
              help='Output format: json or text.')
@click.option('--mock-llm/--no-mock-llm', default=True,
              help='Use Mock LLM for testing (default: True).')
@click.option('--api-key', type=str, default=None,
              help='OpenAI API Key. Reads from OPENAI_API_KEY environment variable if not provided.')
def decompose(goal: str, output: str, mock_llm: bool, api_key: str):
    """
    Decomposes a high-level goal into a series of steps.
    """
    logger.info(f"CLI: 开始分解目标: {goal}")
    llm_client: BaseLLMClient = None

    try:
        if mock_llm:
            llm_client = MockLLMClient()
            logger.info("CLI: 使用 Mock LLM 客户端。")
        else:
            if not api_key:
                api_key = os.getenv("OPENAI_API_KEY")
                if not api_key:
                    raise click.ClickException("OpenAI API Key未提供。请使用 --api-key 选项或设置 OPENAI_API_KEY 环境变量。")
            llm_client = OpenAIClient(api_key=api_key)
            logger.info("CLI: 使用 OpenAI LLM 客户端。")

        decomposer = TaskDecomposer(llm_client=llm_client)
        graph, steps_details = decomposer.decompose_task(goal)

        if output == 'json':
            result = {
                "goal": goal,
                "graph_nodes": [{"id": node, **graph.nodes[node]} for node in graph.nodes()],
                "graph_edges": [[u, v] for u, v in graph.edges()],
                "steps_details": steps_details
            }
            click.echo(json.dumps(result, indent=2, ensure_ascii=False))
        else: # text format
            click.echo(f"目标: {goal}\n")
            click.echo("分解步骤:")
            
            # 拓扑排序以更好地展示依赖关系
            try:
                sorted_nodes = list(nx.topological_sort(graph))
            except nx.NetworkXUnfeasible:
                logger.warning("图包含循环，无法进行拓扑排序。按原始顺序显示。")
                sorted_nodes = list(graph.nodes()) # 如果有循环，则按原始顺序显示

            for node_id in sorted_nodes:
                details = steps_details.get(node_id, {})
                description = details.get("description", "无描述")
                dependencies = details.get("dependencies", [])
                instructions = details.get("instructions", [])

                click.echo(f"\n  ID: {node_id}")
                click.echo(f"  描述: {description}")
                if dependencies:
                    click.echo(f"  依赖: {', '.join(dependencies)}")
                if instructions:
                    click.echo("  指令:")
                    for instr in instructions:
                        click.echo(f"    - {instr}")
            
            click.echo("\n图结构 (节点 -> 依赖):")
            for u, v in graph.edges():
                click.echo(f"  {u} -> {v}")

    except ValueError as e:
        logger.error(f"CLI: 任务分解失败 (输入或逻辑错误): {e}")
        raise click.ClickException(f"任务分解失败 (输入或逻辑错误): {e}")
    except click.ClickException as e:
        logger.error(f"CLI: 命令行参数错误: {e}")
        raise e # ClickException 会自动处理退出码
    except Exception as e:
        logger.critical(f"CLI: 发生意外错误: {e}", exc_info=True)
        raise click.ClickException(f"发生意外错误: {e}")

if __name__ == '__main__':
    cli()