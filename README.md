# CHD Failure Prediction

Проект предназначен для прогнозирования отказов сетевого оборудования на основе телеметрических показателей. Данные собирались с устройств Eltex и Terra, а также предоставлены сведения об инцидентах.

## Структура репозитория

- `metrics/` – сырые телеметрические файлы (`*.txt.gz`).
- `incidents/` – сведения об отказах оборудования.
- `info/` – справочник для расшифровки идентификаторов метрик.
- `preprocessed_data/` – примеры подготовленных датасетов и файлов сопоставления признаков.
- `models/` – директория для сохранения обученных моделей.
- `notebooks/` – исходные исследовательские ноутбуки (в репозитории представлен `test_chd.ipynb`).
- `scripts/` – консольные скрипты для построения полноценного пайплайна.

## Скрипты

### `scripts/make_dataset.py`
Создание набора данных из сырых файлов.

Основные параметры:
- `--metrics-glob` – путь/шаблон к сырым `*.txt.gz`;
- `--incidents` – CSV с инцидентами (необязателен в режиме `--is-test`);
- `--info` – CSV со справочником метрик;
- `--out` – итоговый CSV файл;
- `--feature-dir` – папка для сохранения `feature_list.json` и `rename_map.json`;
- `--time-lag` – горизонт маркировки исходного таргета (по умолчанию 24 часа);
- `--is-test` – обработка тестовых данных без таргета.

### `scripts/train_model.py`
Обучение модели LightGBM.

Основные параметры:
- `--data` – подготовленный CSV из предыдущего шага;
- `--incidents` – файл инцидентов (используется для вычисления `hours_to_fail`);
- `--lag-hours` – желаемый горизонт прогнозирования (формирование таргета);
- `--date-split` – дата раздела на train/test;
- `--device-tag` – суффикс имени сохраняемой модели.

После обучения модель и подобранный порог сохраняются в папку `models/`.

### `scripts/predict.py`
Инференс обученной модели на тестовом наборе данных.

Параметры:
- `--data` – CSV файл с тестовыми признаками;
- `--model` – путь к сохранённой модели (`.txt`);
- `--threshold` – файл с числом‑порогом (`*_thr.txt`);
- `--out` – файл, куда сохраняются предсказания (`ip_address`, `timestamp`, `y_pred`).

## Пример использования
```bash
# Создание датасета
python scripts/make_dataset.py \
    --metrics-glob "metrics/vtb_export_*/*Eltex1.txt.gz" \
    --incidents incidents/incidents_union.csv \
    --info info/info_union.csv \
    --out preprocessed_data/eltex1_24h.csv \
    --feature-dir preprocessed_data/eltex1

# Обучение модели (горизонт 3 часа, раздел по дате)
python scripts/train_model.py \
    --data preprocessed_data/eltex1_24h.csv \
    --incidents incidents/incidents_union.csv \
    --lag-hours 3 \
    --date-split 2024-12-20 \
    --device-tag Eltex1_3h

# Инференс
python scripts/predict.py \
    --data preprocessed_data/eltex1_test.csv \
    --model models/best_lgbm_clf_Eltex1_3h.txt \
    --threshold models/best_lgbm_clf_Eltex1_3h_thr.txt \
    --out predictions_eltex1.csv
```

## Требования
Необходимы Python пакеты из `requirements.txt`. Установка:
```bash
pip install -r requirements.txt
```

