import asyncio
import json
import logging
import os
from datetime import datetime

import requests
import websockets
from websockets import ConnectionClosed

from config import get_data_from_json
from main_cv_utility_module import MainUtility
from os.path import exists
from pathlib import Path
import tempfile

connected_clients = set()


def setup_logging():
    log_dir = 'logs'
    # Создаем директорию для логов, если она не существует
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Удаляем старые файлы логов, которые не относятся к текущему дню
    today = datetime.now().strftime('%d-%m')
    for filename in os.listdir(log_dir):
        if not filename.startswith(today):
            os.remove(os.path.join(log_dir, filename))

    # Настройка логгера
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    file_handler = logging.FileHandler(os.path.join(log_dir, f'{today}_log.txt'))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)


async def handle_websocket(websocket, path):
    connected_clients.add(websocket)
    try:
        async for message in websocket:
            ...
    except websockets.ConnectionClosed:
        pass
    finally:
        # Удаляем клиента из списка при отключении
        connected_clients.remove(websocket)


async def broadcast_message(message):
    # Отправляем сообщение всем подключенным клиентам
    for client in connected_clients:
        try:
            await client.send(message)
        except websockets.ConnectionClosed:
            pass


def action_on_detected_plate_number_method(plate_number: str):
    domain = get_data_from_json('domain')
    endpoint = "cv_assist/plate_number"
    is_depend = get_data_from_json('is_depend')
    url = f"https://{domain}/api/{endpoint}"

    request_data = {
        "plate_number": plate_number,
        "is_depend": is_depend
    }
    result = requests.post(url, json=request_data).json()
    logging.info(f'Запрос на ассистирование с гос.номером {plate_number} отправлен на сервер')
    if 'data' in result:
        data = result['data']
        logging.info(f'Ответ получен: {data}')
        asyncio.run(broadcast_message(json.dumps(result['data'])))


# def create_photo_method(image_base64: str):
#     domain = get_data_from_json('domain')
#     endpoint = "photo/create"
#     url = f"https://{domain}/api/{endpoint}"
#     print(url)
#     request_data = {
#         "photo_bytes": image_base64,
#     }
#     result = requests.post(url, json=request_data)
#     print(result)
#     logging.info(f'Запрос на создание фото отправлен на сервер')
#     if result:
#         logging.info(f'Ответ: status: {result.status_code}, request endpoint: {result.url}, content: {result.content}')
#     else:
#         logging.info(f'Ответа нет')

def create_photo_method(photo_abs_path: str):
    images_folder = os.path.join(str(Path.cwd()), "images")
    if not os.path.exists(images_folder):
        os.mkdir(images_folder)
    domain = get_data_from_json('domain')
    endpoint = "photo/upload"
    url = f"https://{domain}/api/{endpoint}"
    try:
        with open(photo_abs_path, 'rb') as file:
            files = {'file': ('filename.png', file, 'image/png')}
            result = requests.post(url, files=files)
            logging.info(f'Запрос на создание фото отправлен на сервер')
            if result:
                logging.info(f'Ответ: status: {result.status_code}, request endpoint: {result.url}, content: {result.text}')
            else:
                logging.info(f'Произошла ошибка, ответ от сервера: {result.text}')
    except FileNotFoundError:
        logging.info(f'Директории images не существует')
    try:
        os.remove(photo_abs_path)
        logging.info(f"Фото {photo_abs_path} успешно удалено")
    except Exception as e:
        logging.info(f"Фото {photo_abs_path} не удалось удалить, по причине: {e}")


async def start_server():
    logging.info('Запуск Утилиты CV')
    setup_logging()
    weighing_websocket = "ws://localhost:8888"

    util = MainUtility(
        action_on_detected_plate_number_method=action_on_detected_plate_number_method,
        create_photo_method=create_photo_method
    )
    while True:
        try:
            logging.info(f'Попытка подключения к весовой утилите')
            async with websockets.connect(weighing_websocket) as ws:
                logging.info(f'Подключение к весовой утилите успешно')

                async with websockets.serve(handle_websocket, "localhost", 9000):
                    while True:
                        util.last_weight_value = int(await ws.recv())
        except ConnectionRefusedError:
            logging.info(f'Попытка подключения к весовой утилите безуспешна')
            await asyncio.sleep(get_data_from_json('connect_interval'))
            continue
        except ConnectionClosed:
            logging.info(f'Подключение к весовой утилите потеряно')
            util.last_weight_value = 0
            continue


asyncio.run(start_server())
