import pickle
from model import Model


def serialize_model(model):
  with open('model.pkl', 'wb') as f:
    pickle.dump(obj=model, file=f)
  
if __name__ == '__main__':
  model = Model()
  serialize_model(model=model)
  with open('model.pkl', 'rb') as f:
    model2 = pickle.load(file=f)
  print(model2.get_response("request"))

