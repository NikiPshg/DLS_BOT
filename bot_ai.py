import logging
import asyncio
from aiogram import Bot, Dispatcher, executor, types
import torch
from torch.jit import load
import torchvision
import torchvision.transforms as tt
from PIL import Image
import loguru
from loguru import logger
import io


logger.add('logger.log' , format ="{time} {level} :  {message}" , level = "INFO" )

API_TOKEN = '###'

logging.basicConfig(level=logging.INFO)

bot = Bot(token=API_TOKEN)
dp = Dispatcher(bot)


transform_low = tt.transforms.Compose([
    tt.Resize([128,128]),
    tt.ToTensor(),
])

model_path = "D:\python\DLS\generator_cpu.pt"

model = torch.jit.load(model_path)

model.eval()

@dp.message_handler(commands=['start', 'help'])
async def send_welcome(message: types.Message):
    await message.reply("Привет! Я простой телеграм-бот. Отправь мне фотографию, и я её обработаю.")


@dp.message_handler(content_types=types.ContentType.PHOTO)
async def handle_photo(message: types.Message):
    # Получаем информацию о фотографии
    photo = message.photo[ -1 ]
    file_id = photo.file_id
    file_path = await bot.get_file(file_id)
    
    downloaded_file = await bot.download_file(file_path.file_path)

    img_low = Image.open(downloaded_file)

    img_low = transform_low(img_low)

    img_low.cpu().permute(1, 2, 0)

    img_high = model( img_low.unsqueeze(0) ).resize(3,256,256)

    image = tt.ToPILImage()(img_high)

    bio = io.BytesIO()

    image.save(bio, format='PNG')  # Сохраняем изображение в формате PNG

    # Установка указателя на начало потока
    bio.seek(0)

    logger.info(f'Фото отправлено пользователю {message.chat.id}')
    

    await bot.send_photo(message.chat.id, photo=bio)

if __name__ == '__main__':
    executor.start_polling(dp, skip_updates=True)


logger.critical(f'БОТ СЛОМАЛСЯ!!!!')
