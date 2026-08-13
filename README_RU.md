# InferenceRuntime

[English version](README.md)

Экспериментальный runtime для запуска языковых моделей на CPU, написанный на C++20. Сейчас реализована загрузка архитектуры **SmolLM2** из Hugging Face-совместимого каталога моделей и вычисления в `float32`/`float16` с OpenMP и AVX2-ускорением.

> Проект находится в разработке: текущая команда выполняет префилл и декодирование, выводит среднее время токена и скорость, но пока не возвращает сгенерированный текст.

## Требования

- CMake 3.25.1 или новее
- Компилятор с поддержкой C++20
- OpenMP (включён по умолчанию и обязателен для стандартной конфигурации)
- Процессор с AVX2, FMA и F16C для ускоренной сборки

## Сборка

```powershell
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

## Тесты

```powershell
ctest --test-dir build -C Release --output-on-failure
```

## Бенчмарки

Сконфигурируйте проект с `-DBUILD_BENCHMARKS=ON`, затем запустите:

```powershell
.\build\Release\RuntimeBenchmark.exe smollm2 C:\models\SmolLM2-135M "Hello, world!"
```

`RuntimeBenchmark` выполняет пять итераций генерации по 300 токенов и выводит время и токенов в секунду.

Если OpenMP или AVX2 недоступны, их можно отключить при конфигурации:

```powershell
cmake -S . -B build -DENABLE_OPENMP=OFF -DENABLE_AVX2=OFF
```

## Запуск

Передайте архитектуру, путь к каталогу модели и промпт:

```powershell
.\build\Release\InferenceRuntime.exe smollm2 C:\models\SmolLM2-135M "Hello, world!"
```

В каталоге модели должны находиться файлы:

```text
config.json
model.safetensors
tokenizer.json
```

Поддерживается только идентификатор архитектуры `smollm2`. Загрузчик ожидает веса в одном файле `model.safetensors` и тензоры в формате `float16`.

## Устройство проекта

- `src/Backend/CPU` — CPU-бэкенд и операции над тензорами
- `src/Math` — вычислительные ядра, включая AVX2-варианты
- `src/Model/SmolLM2` — загрузка весов, токенизатор и реализация SmolLM2
- `src/Runtime` — высокоуровневый API инференса
- `thirdparty/simdjson` — встроенный парсер JSON

## Ограничения

- Только CPU; число потоков в демонстрационном приложении сейчас задано в `src/Main.cpp`.
- Сэмплирование — greedy (`argmax`).
