import joblib
import cv2
import numpy as np
import librosa
import onnxruntime as rt

# Feature extraction function
def features_extractor(file):
    audio, sample_rate = librosa.load(file, res_type='linear')
    mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_mels=442)
    mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
    return mel_spectrogram

# Pipeline for prediction
def predict_audio_class(audio_file):
    # Load the trained components
    labelencoder = joblib.load('saved_models/labelencoder.pkl')
    scaler = joblib.load('saved_models/scaler.pkl')
    pca = joblib.load('saved_models/pca.pkl')

    # Extract features from the audio file
    features = features_extractor(audio_file)
    features_resized = cv2.resize(features, (224, 224))
    features_flat = features_resized.flatten().reshape(1, -1)
    features_scaled = scaler.transform(features_flat)
    features_pca = pca.transform(features_scaled)

    # Initialize ONNX runtime session
    sess = rt.InferenceSession('saved_models/sklearn_audio_classification_model.onnx', providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    input_shape = sess.get_inputs()[0].shape  # Typically [None, feature_size]

    # Adjust feature size to match model's expected size
    expected_feature_size = input_shape[1]
    if features_pca.shape[1] < expected_feature_size:
        # Pad if feature vector is too short
        padded_features = np.zeros((1, expected_feature_size))
        padded_features[0, :features_pca.shape[1]] = features_pca[0]
    elif features_pca.shape[1] > expected_feature_size:
        # Truncate if feature vector is too long
        padded_features = features_pca[0, :expected_feature_size].reshape(1, -1)
    else:
        padded_features = features_pca

    # Run inference
    label_name = sess.get_outputs()[0].name
    pred_onx = sess.run([label_name], {input_name: padded_features.astype(np.float32)})[0]

    # Debugging: Check the shape of pred_onx
    print(f"ONNX model output (pred_onx): {pred_onx}, Shape: {pred_onx.shape}")

    # Decode prediction
    if pred_onx.ndim == 1:
        # If the output is one-dimensional, use it directly
        prediction_class = labelencoder.inverse_transform(pred_onx.astype(int))
    else:
        # Otherwise, use argmax to get the most probable class
        prediction_class = labelencoder.inverse_transform(pred_onx.argmax(axis=1))

    return prediction_class

# Example usage
