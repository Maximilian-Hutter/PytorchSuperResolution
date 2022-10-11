from PIL import Image
import glob


def crop_center(pil_img, crop_width, crop_height):
    img_width, img_height = pil_img.size
    return pil_img.crop(((img_width - crop_width) // 2,
                         (img_height - crop_height) // 2,
                         (img_width + crop_width) // 2,
                         (img_height + crop_height) // 2))

imgs = sorted(glob.glob("D:/Data/div2k/DIV2K_train_HR" + "/*.*"))

for i, imgpath in enumerate(imgs, start = 1):
    img = Image.open(imgpath)
    img = crop_center(img, 1024, 1024)  # prepare data to 1024x1024 pic
    img.save("D:/Data/div2k_sizecorrected/" + str(i) + ".png")
    print(str(i))
