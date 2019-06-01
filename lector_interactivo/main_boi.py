import pickle
from red2 import Perceptron, Layer, Sample
from prompt_drawing import prompt_draw
from convertir import img_to_array, array_to_sample
from decimal import Decimal

prompt_draw()
img_try_array = img_to_array("img_try.png")
img_try_sample = array_to_sample(img_try_array)
with open("./lector_digits.pkl", "br") as fh:
    perceptron_lector = pickle.load(fh)
resultado = list(perceptron_lector(img_try_sample))
print(resultado.index(max(resultado)))
confianza = []
for i in resultado:
    confianza += [round(Decimal(str(float(i))), 3)]

for i in range(len(confianza)):
    print(str(i)+": "+str(confianza[i]))
