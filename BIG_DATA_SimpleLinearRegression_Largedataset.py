from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import FloatType

# Create a Spark session
spark = SparkSession.builder.appName("SimpleLinearRegressionTaxiDataLarge").getOrCreate()

# Define file path for the large dataset
large_dataset = "gs://met-cs-777-data/taxi-data-sorted-large.csv.bz2"

# Function to check if a value is a float
def isfloat(value):
    try:
        float(value)
        return True
    except ValueError:
        return False

# Function to filter out incorrect rows
def correctRows(p):
    if len(p) == 17:
        if isfloat(p[5]) and isfloat(p[11]):
            trip_distance = float(p[5])
            fare_amount = float(p[11])
            if 0 < trip_distance < 600 and 0 < fare_amount < 600:
                return True
    return False

# Load and preprocess the data
def load_and_preprocess_data(file_path):
    data = spark.read.format("csv").options(header="false", inferSchema="true", sep=",").load(file_path)
    rdd = data.rdd.map(tuple)
    filtered_rdd = rdd.filter(correctRows)
    return filtered_rdd.toDF(["col_" + str(i) for i in range(17)])

# Main execution
if __name__ == "__main__":
    # Load and preprocess data
    df = load_and_preprocess_data(large_dataset)

    # Correct column names for regression
    df = df.withColumn("trip_distance", df["col_5"].cast(FloatType()))
    df = df.withColumn("fare_amount", df["col_11"].cast(FloatType()))

    # Calculate the required sums for the formulas
    N = df.count()  # Total number of records
    sum_x = df.agg(F.sum("trip_distance")).collect()[0][0]
    sum_y = df.agg(F.sum("fare_amount")).collect()[0][0]
    sum_x_squared = df.withColumn("x_squared", F.col("trip_distance") ** 2).agg(F.sum("x_squared")).collect()[0][0]
    sum_xy = df.withColumn("xy", F.col("trip_distance") * F.col("fare_amount")).agg(F.sum("xy")).collect()[0][0]

    # Compute the slope (m) and intercept (b) using the formulas
    m = (N * sum_xy - sum_x * sum_y) / (N * sum_x_squared - sum_x ** 2)
    b = (sum_y - m * sum_x) / N

    # Print the final results
    print(f"Slope (m): {m}")
    print(f"Intercept (b): {b}")

    # Save the results
    output_path = "gs://cs777fall2024_task1hw3smalldatasetsimpleregression/results.json"
    df_result = spark.createDataFrame([(m, b)], ["slope", "intercept"])
    df_result.coalesce(1).write.mode("overwrite").json(output_path)

    # Stop the Spark session
    spark.stop()
