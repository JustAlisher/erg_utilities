import asyncio
import json
import logging
import os
import random
from datetime import datetime

import serial
import websockets
from serial import SerialException

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
    except websockets.ConnectionClosedError:
        pass
    except websockets.ConnectionClosedOK:
        pass
    finally:
        # Удаляем клиента из списка при отключении
        connected_clients.remove(websocket)


async def broadcast_message(message):
    # Отправляем сообщение всем подключенным клиентам
    for client in connected_clients:
        try:
            await client.send(message)
        except websockets.ConnectionClosedError:
            pass
        except websockets.ConnectionClosedOK:
            pass


async def start_server():
    with open('data.json', 'r') as file:
        data = json.load(file)
    value = data['value']
    while True:
        async with websockets.serve(handle_websocket, "localhost", 8888):
            # index = 1
            # simulation_weight_list = [
            #     0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000,
            #     1000, 1100, 1200, 1300, 1400, 1600, 1800, 2000, 2200, 2200, 2200, 2200, 2200, 2200, 2200, 2200, 2200,
            #     2200, 2200, 2200, 1300, 1000,
            #     900, 800, 700, 600, 500, 400, 300, 200, 100, 0
            # ]
            while True:
                # if index + 1 > len(simulation_weight_list):
                #     index = 0
                # try:
                #     final_value = str(simulation_weight_list[index])
                #     index += 1
                # except:
                #     final_value = str(simulation_weight_list[0])
                #     index = 1
                rand_value = str(random.randint(1500, 15000))
                # print(rand_value)
                await asyncio.sleep(1)
                await broadcast_message(value)


asyncio.run(start_server())
