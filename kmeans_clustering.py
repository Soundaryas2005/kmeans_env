import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def load_data():
    file_path = "C:/Users/gnanu/OneDrive/Desktop/ml internship/kmeans_env/customers.csv"
    try:
        df = pd.read_csv(file_path)
        print(f"✅ Loaded data from: {file_path}")
    except FileNotFoundError:
        print("❌ File not found. Please check the path and filename.")
        exit()
    return df

def preprocess_data(df):
    # Use fixed columns (no need for user input)
    selected_columns = ["Age", "AnnualIncome", "SpendingScore"]
    print(f"\n📊 Using columns for clustering: {selected_columns}")
    return df[selected_columns]

def find_optimal_k(X):
    inertia = []
    k_range = range(1, 11)
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42)
        km.fit(X)
        inertia.append(km.inertia_)

    plt.figure(figsize=(8, 5))
    plt.plot(k_range, inertia, marker='o')
    plt.title("Elbow Method for Optimal k")
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Inertia")
    plt.grid(True)
    plt.show()

def apply_kmeans(X, k):
    model = KMeans(n_clusters=k, random_state=42)
    return model.fit_predict(X), model

def main():
    df = load_data()
    X = preprocess_data(df)

    print("\n📈 Generating Elbow Plot to choose optimal k...")
    find_optimal_k(X)

    # You can manually change the number of clusters here
    k = 5
    print(f"\n✅ Applying KMeans with k = {k}")
    labels, model = apply_kmeans(X, k)
    df['Cluster'] = labels

    # Plot the clusters
    plt.figure(figsize=(8, 6))
    plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=labels, cmap='Set1')
    plt.xlabel(X.columns[0])
    plt.ylabel(X.columns[1])
    plt.title("Customer Segments")
    plt.grid(True)
    plt.show()

    # Save the result
    output_path = "C:/Users/gnanu/OneDrive/Desktop/ml internship/kmeans_env/clustered_customers_output.csv"
    df.to_csv(output_path, index=False)
    print(f"\n📁 Clustering complete. Output saved to:\n{output_path}")

if __name__ == "__main__":
    main()

