
from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware

if __name__ == "__main__":
    # 这是一个标准的 Python 入口点惯用法
    # 当脚本直接运行时 (__name__ == "__main__")，这里的代码会被执行
    # 当通过 python -m YourPackageName 执行 __main__.py 时，__name__ 也是 "__main__"
    import argparse
    import uvicorn
    from .log import Log

    parser = argparse.ArgumentParser(
        description="Start a simple HTTP server similar to http.server."
    )
    parser.add_argument(
        'port',
        metavar='PORT',
        type=int,
        nargs='?', # 端口是可选的
        default=8008,
        help='Specify alternate port [default: 8000]'
    )

    parser.add_argument(
        '--env',
        type=str,
        default='dev', # 默认是开发环境
        choices=['dev', 'prod'],
        help='Set the environment (dev or prod) [default: dev]'
    )

    args = parser.parse_args()

    port = args.port
    print(args.env)
    if args.env == "dev":
        port += 100
        Log.reset_level('debug',env = args.env)
        reload = False
    elif args.env == "prod":
        Log.reset_level('info',env = args.env)# ['debug', 'info', 'warning', 'error', 'critical']
        reload = False
    else:
        reload = False

    # 使用 uvicorn.run() 来启动服务器
    # 参数对应于命令行选项
    uvicorn.run(
        app, # 要加载的应用，格式是 "module_name:variable_name"
        host="0.0.0.0",
        port=port,
        reload=reload  # 启用热重载
    )
