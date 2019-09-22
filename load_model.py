from keras.models import load_model

model = load_model('./cats.model')
print(model.summary())