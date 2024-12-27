import pandas as pd
from loger import logger

def document_loader():
    try:
        df = pd.read_csv("hf://datasets/Bogdan01m/wildguardmix-cleaned/documents.csv")
        df.to_csv('documents.csv', index=False)  # Убедитесь, что индекс не сохраняется
        logger.info('Файл загружен успешно.')
        return df  # Возвращаем DataFrame
    except Exception as e:
        logger.error(f"Ошибка при загрузке файла: {e}")
        return None  # Возвращаем None в случае ошибки

df = document_loader()
if df is not None:
    print(df.head())  # Теперь это будет работать, если df не None
else:
    print("Не удалось загрузить данные.")