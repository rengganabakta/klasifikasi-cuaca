import pickle
import os

print(f"Current working directory: {os.getcwd()}")
print(f"Files in current directory: {os.listdir('.')}")

try:
    with open('model.pkl', 'rb') as file:
        model_dict = pickle.load(file)
    print("\nModel loaded successfully!")
    print(f"Type of loaded model: {type(model_dict)}")
    print(f"Model keys: {model_dict.keys()}")
    
    # Check each component
    print("\nChecking model components:")
    print(f"Model type: {type(model_dict['model'])}")
    print(f"Label encoder type: {type(model_dict['label_encoder'])}")
    print(f"Feature names: {model_dict['feature_names']}")
    
except Exception as e:
    print(f"\nError loading model: {str(e)}")
    print(f"Error type: {type(e)}")
    import traceback
    print(f"Traceback: {traceback.format_exc()}") 