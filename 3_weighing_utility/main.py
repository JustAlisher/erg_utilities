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
    except websockets.ConnectionClosed:
        pass
    # except websockets.ConnectionClosedOK:
    #     pass
    finally:
        # Удаляем клиента из списка при отключении
        connected_clients.remove(websocket)


async def broadcast_message(message):
    # Отправляем сообщение всем подключенным клиентам
    for client in connected_clients:
        try:
            await client.send(message)
        except websockets.ConnectionClosed:
            await client.send("-1")
        except Exception:
            await client.send("-2")


async def start_server():
    setup_logging()
    logging.info('Запуск Весовой Утилиты')

    with open('data.json', 'r') as file:
        data = json.load(file)
    baudrate = data['baudrate']
    port = f"COM{data['port']}"

    while True:
        try:
            indicator = serial.Serial(baudrate=baudrate, port=port)
            logging.info('Установка соединения с весовым индикатором прошла успешно')

            async with websockets.serve(handle_websocket, "localhost", 8888):
                finalWeightValue = 0
                while True:
                    # rand_value = str(random.randint(1, 10))
                    # print(rand_value)
                    # await asyncio.sleep(1)
                    # await broadcast_message(rand_value)
                    if not indicator:
                        await broadcast_message("-1")
                        raise SerialException

                    indicator.flushInput()
                    await asyncio.sleep(0.15)
                    try:
                        y = indicator.in_waiting
                    except:
                        await broadcast_message("-1")
                        logging.info('Индикатор не подключен: нет очереди')
                        # code -1 - 'Индикатор не подключен: нет очереди'
                    if y >= 9:
                        weightValue = indicator.readline(9)
                        finalWeightValue = (weightValue[6] - 48) * 100000 + (weightValue[5] - 48) * 10000 + (
                                weightValue[4] - 48) * 1000 + \
                                           (weightValue[3] - 48) * 100 + (weightValue[2] - 48) * 10 + (
                                                   weightValue[1] - 48) * 1
                        await broadcast_message(str(finalWeightValue))
                    else:
                        await broadcast_message(str(finalWeightValue))
        except SerialException as se:
            logging.info(f'Не удалось установить соединение или соединение с весовым индикатором оборвалось: {se}')
            await asyncio.sleep(data['connect_interval'])
            continue


asyncio.run(start_server())
