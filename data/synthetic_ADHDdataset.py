import numpy as np
import pandas as pd

def generate_basic_features(n_samples=1000, adhd=True):
    np.random.seed(42)
    if adhd:
        data = {
            'fixation_duration': np.random.normal(350, 50, n_samples),
            'saccadic_amplitude': np.random.normal(6, 1, n_samples),
            'saccadic_velocity': np.random.normal(350, 50, n_samples),
            'speech_rate': np.random.normal(120, 15, n_samples),
            'pitch_variability': np.random.normal(35, 10, n_samples),
        }
    else:
        data = {
            'fixation_duration': np.random.normal(300, 50, n_samples),
            'saccadic_amplitude': np.random.normal(5, 1, n_samples),
            'saccadic_velocity': np.random.normal(400, 50, n_samples),
            'speech_rate': np.random.normal(140, 15, n_samples),
            'pitch_variability': np.random.normal(20, 5, n_samples),
        }
    return pd.DataFrame(data)

def generate_advanced_speech_features(n_samples=1000, adhd=True):
    np.random.seed(42)
    if adhd:
        jitter = np.random.normal(1.1, 0.2, n_samples)         # %
        shimmer = np.random.normal(4.0, 0.5, n_samples)        # %
        pause_count = np.random.poisson(10, n_samples)
        avg_pause_duration = np.random.normal(0.6, 0.2, n_samples)
    else:
        jitter = np.random.normal(0.5, 0.15, n_samples)
        shimmer = np.random.normal(2.0, 0.3, n_samples)
        pause_count = np.random.poisson(4, n_samples)
        avg_pause_duration = np.random.normal(0.3, 0.1, n_samples)

    jitter = np.clip(jitter, 0, None)
    shimmer = np.clip(shimmer, 0, None)
    avg_pause_duration = np.clip(avg_pause_duration, 0, None)

    return pd.DataFrame({
        'jitter': jitter,
        'shimmer': shimmer,
        'pause_count': pause_count,
        'avg_pause_duration': avg_pause_duration
    })

def generate_synthetic_adhd_dataset(n_samples=1000):
    df_adhd_basic = generate_basic_features(n_samples, adhd=True)
    df_adhd_advanced = generate_advanced_speech_features(n_samples, adhd=True)
    df_adhd = pd.concat([df_adhd_basic, df_adhd_advanced], axis=1)
    df_adhd['adhd_label'] = 1

    df_non_adhd_basic = generate_basic_features(n_samples, adhd=False)
    df_non_adhd_advanced = generate_advanced_speech_features(n_samples, adhd=False)
    df_non_adhd = pd.concat([df_non_adhd_basic, df_non_adhd_advanced], axis=1)
    df_non_adhd['adhd_label'] = 0

    df = pd.concat([df_adhd, df_non_adhd], ignore_index=True)
    return df

# Generate dataset
df = generate_synthetic_adhd_dataset(n_samples=1000)
print(df.head())

# Save to CSV if needed
df.to_csv("C:/Users/somas/PycharmProjects/ADHD_PREDICTION_MODEL/data/synthetic_adhd_full_dataset.csv", index=False)
print("✅ Synthetic dataset saved as 'synthetic_adhd_full_dataset.csv'")
