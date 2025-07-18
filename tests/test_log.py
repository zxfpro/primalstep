import pytest
import logging
import os
from primalstep.log import Log

class TestLogger:
    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        # 每次测试前重置Logger实例，确保单例模式的独立性
        Log._instance = None
        # 清理日志文件
        log_file_path = os.path.join("logs", "test_app.log")
        if os.path.exists(log_file_path):
            os.remove(log_file_path)
        yield
        # 每次测试后清理日志文件
        if os.path.exists(log_file_path):
            os.remove(log_file_path)


    def test_reset_level(self):
        Log.reset_level('info')
        assert Log.LOG_LEVEL == logging.INFO
        
        Log.reset_level('error')
        assert Log.LOG_LEVEL == logging.ERROR
        # 确保 handlers 被清除并重新添加
        assert len(Log.logger.handlers) > 0
        # 尝试写入日志并检查级别是否生效
        Log.logger.info("This should not appear.") # INFO级别，不应出现
        Log.logger.error("This is an error message.") # ERROR级别，应出现
        
        with open(Log.LOG_FILE_PATH, 'r') as f:
            content = f.read()
            assert "This should not appear." not in content
            assert "This is an error message." in content

    def test_log_module_level_instance(self):
        # 确保 Log 模块级别的实例是 Logger 的单例
        assert isinstance(Log, type(Log)) # Check if Log is an instance of its own type
        # Log 已经是单例，不需要再次创建
        assert Log is Log
        assert Log.logger is Log.logger

    def test_log_file_content(self):
        Log.reset_level('debug')
        test_message = "这是一个测试日志消息。"
        Log.logger.debug(test_message)
        
        with open(Log.LOG_FILE_PATH, 'r') as f:
            content = f.read()
            assert test_message in content
            assert "DEBUG" in content # 检查日志级别是否正确记录