import sklearn
import numpy
import pickle
import os

print(f"scikit-learn version: {sklearn.__version__}")
print(f"numpy version: {numpy.__version__}")

# Check model
try:
    with open('model.pkl', 'rb') as file:
        model_dict = pickle.load(file)
    print("\nModel components:")
    print(f"Model type: {type(model_dict['model'])}")
    print(f"Model version: {model_dict['model'].__class__.__module__}")
    print(f"Label encoder type: {type(model_dict['label_encoder'])}")
    print(f"Label encoder version: {model_dict['label_encoder'].__class__.__module__}")
except Exception as e:
    print(f"Error: {str(e)}") 