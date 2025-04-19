from pyspark.sql import SparkSession
import numpy as np
from pyspark.sql import functions as F
from pyspark.sql.types import FloatType

# Create a Spark session with increased maxResultSize
spark = SparkSession.builder \
    .appName("MultipleLinearRegressionGD") \
    .config("spark.driver.maxResultSize", "4g") \
    .getOrCreate()

# Load and preprocess data
def load_and_preprocess_data(file_path):
    data = spark.read.format("csv").options(header="false", inferSchema="true", sep=",").load(file_path)
    rdd = data.rdd.map(tuple)
    filtered_rdd = rdd.filter(lambda p: len(p) == 17 and all(isfloat(p[i]) for i in [4, 5, 11, 15, 16]) and float(p[4]) > 60)
    return filtered_rdd.map(lambda p: (float(p[4]), float(p[5]), float(p[11]), float(p[15]), float(p[16])))

# Check if value is float
def isfloat(value):
    try:
        float(value)
        return True
    except ValueError:
        return False

# Gradient Descent with Bold Driver
def gradient_descent(data_rdd, learning_rate, max_iterations=50):
    N = data_rdd.count()
    
    # Initialize parameters
    m = np.zeros(4)
    b = 0.0
    
    # Cache the RDD
    data_rdd.cache()
    
    costs = []
    last_cost = float('inf')
    
    for i in range(max_iterations):
        # Calculate predictions and errors for all data points in a distributed way
        gradient_m_b = data_rdd.map(lambda p: (np.dot(np.array(p[:-1]), m) + b - p[-1], np.array(p[:-1])))

        # Calculate gradients in a distributed manner
        sum_gradients = gradient_m_b.map(lambda res: (res[0] * res[1], res[0])).reduce(lambda a, b: (a[0] + b[0], a[1] + b[1]))
        
        # Average gradients
        gradient_m = (1 / N) * sum_gradients[0]
        gradient_b = (1 / N) * sum_gradients[1]

        # Update parameters
        m -= learning_rate * gradient_m
        b -= learning_rate * gradient_b

        # Calculate cost (MSE) in a distributed way
        total_cost = data_rdd.map(lambda p: (np.dot(np.array(p[:-1]), m) + b - p[-1]) ** 2).reduce(lambda a, b: a + b)
        cost = (1 / (2 * N)) * total_cost
        costs.append(cost)
        
        # Bold Driver: Adjust learning rate dynamically
        if cost < last_cost:
            learning_rate = min(learning_rate * 1.10, 1e-2)  # Increase learning rate, but cap it to avoid exploding updates
        else:
            learning_rate = max(learning_rate * 0.7, 1e-6)   # Less aggressive decrease to avoid divergence

        # Early stopping condition
        early_stopping_threshold = 1e-6
        if abs(last_cost - cost) < early_stopping_threshold:
            print("Convergence reached.")
            break
        
        last_cost = cost
        
        print(f"Iteration {i+1}, Cost: {cost}, m: {m}, b: {b}, Learning Rate: {learning_rate}")
    
    return m, b, costs

# Main function
if __name__ == "__main__":
    file_path = "gs://met-cs-777-data/taxi-data-sorted-large.csv.bz2"
    data_rdd = load_and_preprocess_data(file_path)
    
    learning_rate = 0.00001
    max_iterations = 50
    
    # Perform Gradient Descent
    final_m, final_b, cost_history = gradient_descent(data_rdd, learning_rate, max_iterations)
    
    print(f"Final parameters: m = {final_m}, b = {final_b}")
    
    # Save results
    output_path = "gs://cs777fall2024_task3hw3new/gradient_descent_results"
    spark.createDataFrame([(final_m.tolist(), final_b, cost_history)], ["m", "b", "cost_history"]).write.mode("overwrite").json(output_path)
    
    # Stop Spark session
    spark.stop()
