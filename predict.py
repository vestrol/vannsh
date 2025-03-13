import soundfile
import numpy as np
import librosa
import onnxruntime as rt



def extract_feature(file_name, **kwargs):
    """
    Extract feature from audio file `file_name`
    Features supported:
        - MFCC (mfcc)
        - Chroma (chroma)
        - MEL Spectrogram Frequency (mel)
        - Contrast (contrast)
        - Tonnetz (tonnetz)
    e.g:
    `features = extract_feature(path, mel=True, mfcc=True)`
    """
    mfcc = kwargs.get("mfcc")
    chroma = kwargs.get("chroma")
    mel = kwargs.get("mel")
    contrast = kwargs.get("contrast")
    tonnetz = kwargs.get("tonnetz")
    
    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate = sound_file.samplerate
        
        if chroma or contrast:
            stft = np.abs(librosa.stft(X))
            
        result = []
        
        if mfcc:
            mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result.append(mfccs)
            
        if chroma:
            chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
            result.append(chroma)
            
        if mel:
            mel = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T, axis=0)
            result.append(mel)
            
        if contrast:
            contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T, axis=0)
            result.append(contrast)
            
        if tonnetz:
            tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T, axis=0)
            result.append(tonnetz)
    
    # Concatenate all features along the first axis
    result = np.concatenate(result, axis=0)
    return result

def predict_hindi(file_name):
    # all emotions in the new dataset
    int2emotion_new = {
        "anger": "angry",
        "sad": "sad",
        "happy": "happy",
        "neutral": "neutral"
    }

    # we allow only these four emotions
    AVAILABLE_EMOTIONS_NEW = set(int2emotion_new.values())

    features = extract_feature(file_name,mfcc=True,chroma=True,mel=True)
    print(file_name)
    sess = rt.InferenceSession("random_forest_model_with_numpy.onnx", providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    print(input_name)
    input_shape = sess.get_inputs()[0].shape  # Typically [None, feature_size]
    print(input_shape)
    expected_feature_size = input_shape[1]  # Feature size from ONNX model

    # Adjust features to match model's expected size
    if features.shape[0] < expected_feature_size:
        # Pad if feature vector is too short
        padded_features = np.pad(features, (0, expected_feature_size - features.shape[0]))
    elif features.shape[0] > expected_feature_size:
        # Truncate if feature vector is too long
        padded_features = features[:expected_feature_size]
    else:
        padded_features = features

    # Reshape to match ONNX input (batch size = 1)
    padded_features = padded_features.reshape(1, -1).astype(np.float32)

    # Run inference
    label_name = sess.get_outputs()[0].name
    # input_name = sess.get_inputs()[0].name
    # label_name = sess.get_outputs()[0].name

    pred_onx = sess.run([label_name], {input_name: padded_features.astype(np.float32).reshape(1,-1)})[0]
    print(pred_onx)
    return pred_onx[0]

