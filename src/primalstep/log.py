import logging
import os
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler

class Logger:
    """log

    Returns:
        _type_: single
    """
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self,level = 'debug',log_file_name = "app.log"):
        """
        Level (级别): 定义日志的严重程度。从低到高依次是：
        DEBUG: 详细的调试信息，通常只在开发阶段使用。
        INFO: 确认程序按预期运行。
        WARNING: 发生了意外，或将来可能出现问题，但程序仍在正常运行。
        ERROR: 发生了严重错误，程序可能无法执行某些功能。
        CRITICAL: 发生了严重错误，程序可能无法继续运行。
                # --- 1. 定义日志级别 ---
        # 建议在开发环境使用 DEBUG，生产环境使用 INFO 或 WARNING

        """
        if not hasattr(self, 'initialized'):
            self.initialized = True

            if level == 'debug':
                self.LOG_LEVEL = logging.DEBUG  # 开发阶段
            elif level == 'info':
                self.LOG_LEVEL = logging.INFO  # 生产阶段
            elif level == 'warning':
                self.LOG_LEVEL = logging.WARNING
            elif level == 'error':
                self.LOG_LEVEL = logging.ERROR
            elif level == 'critical':
                self.LOG_LEVEL = logging.CRITICAL
            else:
                self.LOG_LEVEL = logging.INFO  # 默认级别

            # --- 2. 定义日志文件路径和名称 ---
            self.LOG_DIR = "logs"
            self.LOG_FILE_NAME = log_file_name
            self.LOG_FILE_PATH = os.path.join(self.LOG_DIR, self.LOG_FILE_NAME)

            # 确保日志目录存在
            os.makedirs(self.LOG_DIR, exist_ok=True)
            self.logger = None
            self.setup_logging()
            self.env = 'dev'

    def reset_level(self,level = 'debug',env = 'dev'):
        if level == 'debug':
            self.LOG_LEVEL = logging.DEBUG  # 开发阶段
        elif level == 'info':
            self.LOG_LEVEL = logging.INFO  # 生产阶段
        elif level == 'warning':
            self.LOG_LEVEL = logging.WARNING
        elif level == 'error':
            self.LOG_LEVEL = logging.ERROR
        elif level == 'critical':
            self.LOG_LEVEL = logging.CRITICAL
        else:
            self.LOG_LEVEL = logging.INFO  # 默认级别

        self.setup_logging()
        self.env = env

    def setup_logging(self):
        """# --- 3. 配置 Logger ---
        """
        # 获取根 Logger (也可以创建自定义的 Logger: logging.getLogger('my_app'))
        logger = logging.getLogger()
        logger.setLevel(self.LOG_LEVEL)

        # 避免重复添加 Handler (如果多次调用 setup_logging)
        if not logger.handlers:
            # --- 4. 配置 Formatter (格式化器) ---
            # 常见格式：时间 - 日志级别 - Logger名称 - 模块名 - 行号 - 消息
            formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s - %(name)s - %(module)s:%(lineno)d - %(message)s'
            )

            # --- 5. 配置 Handler (处理器) ---

            # 5.1 控制台处理器 (StreamHandler)
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO) # 控制台只显示 INFO 及以上级别的日志
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)

            # 5.2 文件处理器 (RotatingFileHandler 或 TimedRotatingFileHandler)

            # RotatingFileHandler: 按文件大小轮转
            # maxBytes: 单个日志文件的最大字节数 (例如 10MB)
            # backupCount: 保留的旧日志文件数量
            file_handler = RotatingFileHandler(
                self.LOG_FILE_PATH,
                maxBytes=10 * 1024 * 1024, # 10 MB
                backupCount=5,
                encoding='utf-8'
            )
            file_handler.setLevel(self.LOG_LEVEL) # 文件中显示所有指定级别的日志
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

            # TimedRotatingFileHandler: 按时间轮转 (例如每天轮转)
            # when='midnight': 每天午夜轮转
            # interval=1: 每隔1个单位轮转
            # backupCount: 保留的旧日志文件数量 (例如保留最近7天的日志)
            # file_handler = TimedRotatingFileHandler(
            #     LOG_FILE_PATH,
            #     when='midnight',
            #     interval=1,
            #     backupCount=7,
            #     encoding='utf-8'
            # )
            # file_handler.setLevel(LOG_LEVEL)
            # file_handler.setFormatter(formatter)
            # logger.addHandler(file_handler)
        self.logger = logger


Log = Logger(log_file_name = "app.log")
del Logger
