from pyspark.sql import SparkSession
from pyspark import SparkContext
import sys

# Create a Spark session and context
spark = SparkSession.builder.appName("GradientDescentTaxiDataLarge").getOrCreate()
sc = spark.sparkContext

# Function to check if a value is float
def isfloat(value):
    try:
        float(value)
        return True
    except ValueError:
        return False

# Function to clean and filter incorrect rows
def correctRows(p):
    if len(p) == 17:
        if isfloat(p[5]) and isfloat(p[11]):
            trip_distance = float(p[5])
            fare_amount = float(p[11])
            if 0 < trip_distance < 6000 and 0 < fare_amount < 6000:
                return True
    return False

# Load and preprocess the data
def load_and_preprocess_data(file_path):
    data = spark.read.format("csv").options(header="false", inferSchema="true", sep=",").load(file_path)
    rdd = data.rdd.map(tuple)
    filtered_rdd = rdd.filter(correctRows)
    return filtered_rdd

# Extract the necessary columns for regression (trip_distance and fare_amount)
def get_x_y(rdd):
    return rdd.map(lambda p: (float(p[5]), float(p[11])))

# Gradient Descent function to calculate the cost and update model parameters
def batch_gradient_descent(xy_rdd, learning_rate=0.0001, num_iterations=50):
    N = xy_rdd.count()

    # Initialize m (slope) and b (intercept)
    m, b = 0.0, 0.0

    # List to store the costs for each iteration
    costs = []

    for i in range(num_iterations):
        # Compute the gradients
        gradients = xy_rdd.map(lambda p: (
            p[0] * (m * p[0] + b - p[1]),  # Partial derivative w.r.t m
            (m * p[0] + b - p[1])          # Partial derivative w.r.t b
        )).reduce(lambda a, b: (a[0] + b[0], a[1] + b[1]))

        # Update the parameters m and b
        m -= (learning_rate * gradients[0]) / N
        b -= (learning_rate * gradients[1]) / N

        # Calculate the cost (loss function)
        loss = xy_rdd.map(lambda p: (m * p[0] + b - p[1]) ** 2).sum() / N
        costs.append(loss)

        # Print the cost and parameters at each iteration
        print(f"Iteration {i + 1}: Loss = {loss:.4f}, m = {m:.6f}, b = {b:.6f}")

    return m, b, costs

# Function to save results to GCS
def save_results(output_path, m, b, costs):
    from pyspark.sql import Row

    # Create DataFrame with the final results and costs
    results = [{"iteration": idx + 1, "cost": cost, "slope_m": m, "intercept_b": b} for idx, cost in enumerate(costs)]
    results_df = spark.createDataFrame(results)

    # Save the DataFrame as a CSV file in GCS
    results_df.coalesce(1).write.mode("overwrite").csv(output_path)

# Main function to execute the job
if __name__ == "__main__":
    # Check the arguments
    if len(sys.argv) != 3:
        print("Usage: gradient_descent <input_file> <output_path>", file=sys.stderr)
        exit(-1)

    # Load and preprocess data
    input_file = sys.argv[1]
    output_path = sys.argv[2]

    data_rdd = load_and_preprocess_data(input_file)
    xy_rdd = get_x_y(data_rdd)
    xy_rdd = xy_rdd.cache()  # Cache the RDD for optimization

    # Set the learning rate and number of iterations for the large dataset
    learning_rate = 0.0001
    num_iterations = 50

    # Run gradient descent to calculate m (slope) and b (intercept)
    m, b, costs = batch_gradient_descent(xy_rdd, learning_rate, num_iterations)

    # Print the final results
    print(f"\nFinal parameters:\nSlope (m): {m}\nIntercept (b): {b}")

    # Save the results to GCS
    save_results(output_path, m, b, costs)

    # Stop the Spark session
    spark.stop()
