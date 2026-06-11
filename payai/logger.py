import logging

# Configuracao basica de logging usada pelo aplicativo
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('payai.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('payai')
