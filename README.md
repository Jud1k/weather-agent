# Weather Agent 🌤️

Умный AI-агент для получения информации о погоде в различных городах мира. Проект построен на современном стеке технологий с использованием LangGraph для управления workflow и FastMCP для взаимодействия с внешними API.

## 🚀 Возможности

- **Интеллектуальный анализ запросов** - автоматическое определение города из текстового запроса
- **Интеграция с WeatherAPI** - получение актуальных данных о погоде
- **Визуальное управление workflow** - четкое разделение логики на узлы графа
- **Асинхронная обработка** - высокая производительность и отзывчивость
- **MCP-архитектура** - модульность и легкое расширение функционала

## 🛠️ Технологический стек

- **Python 3.12** - основной язык разработки
- **LangGraph** - оркестрация workflow агента
- **FastMCP** (Model Context Protocol) - взаимодействие с внешними API
- **GigaChat** - языковая модель для анализа запросов
- **Pydantic v2** - валидация данных и моделей
- **HTTPX** - асинхронные HTTP-запросы
- **UV** - современный менеджер пакетов и зависимостей

## 📦 Установка и запуск

### Предварительные требования

- Python 3.11+
- UV package manager
- API ключ от [GigaChat](https://developers.sber.ru/studio/login)
- API ключ от [WeatherAPI](https://www.weatherapi.com/)

### Клонирование и настройка

```bash
# Клонирование репозитория
git clone https://github.com/your-username/travel-weather-agent.git
cd travel-weather-agent

# Создание виртуального окружения и установка зависимостей
uv venv
source .venv/bin/activate  # Linux/MacOS
# или
.venv\Scripts\activate     # Windows

uv pip install -r requirements.txt

