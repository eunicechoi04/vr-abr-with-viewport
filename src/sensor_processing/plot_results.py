import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_accuracy_by_video(csv_path):
    df = pd.read_csv(csv_path)
    
    # Clean up video names (remove the unique ID part for cleaner labels)
    # e.g. "Diving-2OzlksZBTiA" -> "Diving"
    df['Video'] = df['parent_folder'].apply(lambda x: x.split('-')[0])
    
    # 1. Box Plot (Shows the spread/variance)
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Video', y='accuracy', data=df, palette="Set3")
    plt.title("Viewport Prediction Accuracy by Video Content")
    plt.ylabel("Hit Rate (0.0 - 1.0)")
    plt.ylim(0, 1.0)
    plt.grid(True, alpha=0.3)
    plt.savefig("graph_prediction_accuracy.png")
    print("Saved graph_prediction_accuracy.png")
    
    # 2. Print Average per Video
    print("\nAverage Accuracy per Video:")
    print(df.groupby('Video')['accuracy'].mean())

if __name__ == "__main__":
    plot_accuracy_by_video("src/final_prediction_results.csv")