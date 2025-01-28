# main.py
from src.config import Config
from src.data_processor import DataProcessor
from src.model_trainer import ModelTrainer
from src.model_evaluator import ModelEvaluator
from src.user_interface import UserInterface
from src.predictor import Predictor
import os

def train_models():
    config = Config()
    data_processor = DataProcessor(config)
    model_trainer = ModelTrainer(config)
    
    df = data_processor.load_data('data/diabetes.csv')
    X_train, X_test, y_train, y_test = data_processor.preprocess_data(df)
    trained_models = model_trainer.train_models(X_train, y_train)
    
    return trained_models, X_test, y_test

def predict_for_user():
    config = Config()
    user_interface = UserInterface(config)
    data_processor = DataProcessor(config)
    
    user_data = user_interface.get_user_input()
    processed_features = data_processor.preprocess_single_sample(user_data)
    prediction, probability = Predictor.predict(processed_features)
    user_interface.display_prediction(prediction, probability)

def main():
    print("=== Diabetes Prediction System ===")
    
    if not os.path.exists('models/saved_models/random_forest_model.pkl'):
        print("\nTraining models for the first time...")
        trained_models, X_test, y_test = train_models()
        
        evaluator = ModelEvaluator()
        results = evaluator.evaluate_models(trained_models, X_test, y_test)
        
        print("\nModel Training Complete!")
        print("\nModel Accuracies:")
        for name, metrics in results.items():
            print(f"{name.capitalize()}: {metrics['accuracy']:.4f}")
    
    while True:
        predict_for_user()
        choice = input("\nWould you like to make another prediction? (y/n): ")
        if choice.lower() != 'y':
            break

if __name__ == "__main__":
    main()