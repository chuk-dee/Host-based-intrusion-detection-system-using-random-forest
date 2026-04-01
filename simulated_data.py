Generate Simulated Data for Testing
import csv
import random
import time

with open('sensor_data.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['temperature', 'gyro_x', 'gyro_y', 'gyro_z', 'timestamp', 'label'])

    # Generate 500 normal
    for _ in range(500):
        temp = random.uniform(20, 30)
        gx = random.uniform(-0.1, 0.1)
        gy = random.uniform(-0.1, 0.1)
        gz = random.uniform(-0.1, 0.1)
        ts = time.time()
        writer.writerow([temp, gx, gy, gz, ts, 0])

    # Generate 500 anomalies
    for _ in range(500):
        temp = random.uniform(50, 90)
        gx = random.uniform(-2, 2)
        gy = random.uniform(-2, 2)
        gz = random.uniform(-2, 2)
        ts = time.time()
        writer.writerow([temp, gx, gy, gz, ts, 1])