from PIL import Image, ImageDraw, ImageFont, ImageFilter
import os

<<<<<<< HEAD
font_name = "LibreFranklin-Medium.ttf"  # Make sure this .ttf is in the same folder as this script!
=======
font_name = "LibreFranklin-Medium.ttf"  # I did it with multiple fonts
>>>>>>> 0940acc506f031b5cbae5bd40a505c5b6b1eaf7d
img_size = (98, 100)
target_ratio = 0.8
max_font_size = 200

if not os.path.exists(font_name):
    raise FileNotFoundError(f"Font file '{font_name}' not found! Please put Poppins-Regular.ttf in this directory.")

for digit in range(1, 10):
    font_size = 10
    while True:
        font = ImageFont.truetype(font_name, font_size)
        temp_img = Image.new('L', img_size, color=255)
        draw = ImageDraw.Draw(temp_img)
        try:
            bbox = draw.textbbox((0, 0), str(digit), font=font)
        except AttributeError:
            w, h = font.getsize(str(digit))
            bbox = (0, 0, w, h)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        if (text_w >= img_size[0] * target_ratio or text_h >= img_size[1] * target_ratio) or font_size > max_font_size:
            break
        font_size += 1
<<<<<<< HEAD

    # Center the digit
=======
 
>>>>>>> 0940acc506f031b5cbae5bd40a505c5b6b1eaf7d
    x = (img_size[0] - text_w) // 2 - bbox[0]
    y = (img_size[1] - text_h) // 2 - bbox[1]

    img = Image.new('L', img_size, color=255)
    draw = ImageDraw.Draw(img)
    draw.text((x, y), str(digit), font=font, fill=0)
<<<<<<< HEAD
    img = img.filter(ImageFilter.GaussianBlur(radius=0.5))  # Optional: blur for "realistic" effect
=======
    img = img.filter(ImageFilter.GaussianBlur(radius=0.5))    
>>>>>>> 0940acc506f031b5cbae5bd40a505c5b6b1eaf7d
    img.save(f'nr_{digit+126}.png')
