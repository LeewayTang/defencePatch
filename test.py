from PIL import Image, ImageFilter

if __name__ == '__main__':
    im = Image.open(r"stop.jpeg")
    gbF = im.filter(ImageFilter.GaussianBlur(radius=2))
    gbF.show()