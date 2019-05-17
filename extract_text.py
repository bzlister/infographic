from PIL import Image
import pytesseract
import os, os.path

pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract'
base = 'C:\\Users\\bzlis\\Documents\\infographic\\images'
for f in os.listdir(base):
    text = pytesseract.image_to_string(Image.open(base + "\\" + f))
    name = f.split(".")[0]
    outfile = open("text\\" + name + ".txt", 'w')
    outfile.write(text)
    outfile.close()
