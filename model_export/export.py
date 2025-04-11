import joblib
import os

def export_model(model, filename, export_path='exports'):
    """
    Save the trained ML model to a file using joblib.

    Parameters:
    - model: The trained ML model (e.g., RandomForestClassifier)
    - filename: The filename for the saved model (default: trained_model.pkl)
    - export_path: Directory to save the model file

    Returns:
    - Full path to the saved model
    """
    # Create the export directory if it doesn't exist
    if not os.path.exists(export_path):
        os.makedirs(export_path)
    
    model_path = os.path.join(export_path, filename)
    
    # Save the model
    joblib.dump(model, model_path)
    print(f"✅ Model saved to: {model_path}")
    return model_path

# Example usage after training
# trained_model = RandomForestClassifier().fit(X_train, y_train)
# export_model(trained_model, 'rf_model.pkl')
