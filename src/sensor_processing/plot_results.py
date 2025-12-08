import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_accuracy_by_video(csv_path):
    print(f"Looking for results at: {csv_path}")
    if not os.path.exists(csv_path):
        print("Error: File not found. Please check the path.")
        return

    df = pd.read_csv(csv_path)
    
    # Clean up video names (remove the unique ID part for cleaner labels)
    # e.g. "Diving-2OzlksZBTiA" -> "Diving"
    # We use a try/except block in case the format differs
    try:
        df['Video'] = df['parent_folder'].apply(lambda x: x.split('-')[0])
    except:
        df['Video'] = df['parent_folder'] # Fallback
    
    # 1. Box Plot (Shows the spread/variance)
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Video', y='accuracy', data=df, palette="Set3")
    plt.title("Viewport Prediction Accuracy by Video Content")
    plt.ylabel("Hit Rate (0.0 - 1.0)")
    plt.ylim(0, 1.0)
    plt.grid(True, alpha=0.3)
    
    # Save in the same folder as the script
    output_path = os.path.join(os.path.dirname(__file__), "graph_prediction_accuracy.png")
    plt.savefig(output_path)
    print(f"Saved graph to: {output_path}")
    
    # 2. Print Average per Video
    print("\nAverage Accuracy per Video:")
    print(df.groupby('Video')['accuracy'].mean())

if __name__ == "__main__":
    # --- FIX: USE ABSOLUTE PATH ---
    # Based on your screenshots, the CSV is in '360_Video_analysis/src'
    CSV_PATH = r"C:\Users\feido\Documents\Code\6.5820\vr-abr-with-viewport\360_Video_analysis\src\final_prediction_results.csv"
    
    plot_accuracy_by_video(CSV_PATH)