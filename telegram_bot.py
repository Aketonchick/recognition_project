import asyncio

import requests
from aiogram import Bot, Dispatcher
from aiogram.filters import Command, CommandStart
from aiogram.types import Message

# Токен бота от https://t.me/BotFather
TOKEN = "your_token"

# Штука, которая обрабатывает команды бота
dp = Dispatcher()

# Эта функция обрабатывает команду /start
@dp.message(CommandStart())
async def command_start_handler(message: Message) -> None:
    await message.answer(f"Привет! Напиши /random_dog")

async def main() -> None:
    # создается специальный объект бота, который будет общаться с API Telegram
    bot = Bot(token=TOKEN)
    # тут запускается обработка событий того, что происходит с ботом
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())