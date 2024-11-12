import logging

def setup_logger(log_file):
    # Lấy logger mặc định
    logger = logging.getLogger()
    
    # Kiểm tra và xóa các handler cũ
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # Cấu hình lại logger
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )