import json


def get_data_from_json(key):
    file_path = "data.json"
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)  # Загружаем JSON данные из файла
            return data.get(key)
    except FileNotFoundError:
        print(f"Файл '{file_path}' не найден.")
        return None
    except json.JSONDecodeError:
        print(f"Ошибка декодирования JSON данных из файла '{file_path}'.")
        return None
