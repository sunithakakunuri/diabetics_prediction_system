
# predict.py
import joblib
from src.config import Config
from src.predictor import DiabetesPredictor

def get_user_input():
    """Get user details through console input."""
    print("\n=== Diabetes Prediction System ===")
    print("Please enter the following details:")
    
    user_data = {}
    for feature in Config.FEATURE_RANGES.keys():
        while True:
            try:
                value = float(input(f"{feature}: "))
                min_val, max_val = Config.FEATURE_RANGES[feature]
                if min_val <= value <= max_val:
                    user_data[feature] = value
                    break
                else:
                    print(f"Value should be between {min_val} and {max_val}")
            except ValueError:
                print("Please enter a valid number")
    
    return user_data

def main():
    # Initialize configuration and predictor
    config = Config()
    scaler = joblib.load('models/scaler.joblib')
    predictor = DiabetesPredictor(config, scaler)
    
    # Load trained models
    model_paths = {
        'logistic': 'models/logistic_model.joblib',
        'random_forest': 'models/rf_model.joblib',
        'xgboost': 'models/xgb_model.joblib'
    }
    predictor.load_models(model_paths)
    
    # Get user input
    user_data = get_user_input()
    
    # Make predictions
    try:
        predictions = predictor.predict(user_data)
        
        # Display results
        print("\n=== Prediction Results ===")
        for model_name, result in predictions.items():
            print(f"\n{model_name.upper()}:")
            print(f"Prediction: {result['prediction']}")
            print(f"Probability of Diabetes: {result['probability']:.2%}")
        
        # Calculate consensus
        diabetic_votes = sum(1 for result in predictions.values() 
                           if result['prediction'] == 'Diabetic')
        avg_probability = np.mean([result['probability'] 
                                 for result in predictions.values()])
        
        print("\n=== Consensus ===")
        print(f"Models predicting Diabetic: {diabetic_votes}/3")
        print(f"Average probability of Diabetes: {avg_probability:.2%}")
        
        if avg_probability > 0.5:
            print("\nRECOMMENDATION: Please consult a healthcare provider for proper evaluation.")
        else:
            print("\nRECOMMENDATION: Your risk appears to be low, but maintain regular check-ups.")
            
    except ValueError as e:
        print(f"\nError: {str(e)}")
        print("Please try again with valid input values.")

if __name__ == "__main__":
    main()