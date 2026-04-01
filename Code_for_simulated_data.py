import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def generate_vault_hids_dataset(
    num_normal_entries=10000,
    num_anomaly_entries=1000,
    output_filename="dataset.csv"
):
    """
    Generates a synthetic dataset for a host-based intrusion detection system
    on a high-tech vault.

    Args:
        num_normal_entries (int): Number of normal data entries to generate.
        num_anomaly_entries (int): Number of anomalous data entries to generate.
        output_filename (str): Name of the CSV file to save the dataset.
    """

    data = []
    start_time = datetime(2025, 1, 1, 8, 0, 0) # Start from Jan 1, 2025, 8 AM

    authorized_rfids = ["RFID_VAULT_ADMIN_001", "RFID_SECURITY_002", "RFID_MAINTENANCE_003"]
    unauthorized_rfids = ["RFID_UNKNOWN_A", "RFID_UNKNOWN_B", "RFID_BLACKHAT_001"]

    # --- Generate Normal Data ---
    print(f"Generating {num_normal_entries} normal entries...")
    current_time = start_time
    for _ in range(num_normal_entries):
        # Time increment
        current_time += timedelta(seconds=random.randint(5, 60)) # Random interval between 5-60 seconds

        # Temperature: Stable with minor fluctuations
        temp = round(random.uniform(20.0, 25.0), 2) # Normal room temperature

        # Gyro: Mostly zero, very minor sensor noise
        gyro_x = round(random.uniform(-0.01, 0.01), 3)
        gyro_y = round(random.uniform(-0.01, 0.01), 3)
        gyro_z = round(random.uniform(-0.01, 0.01), 3)

        # RFID: Occasional authorized access during "business hours"
        rfid_tag = "NONE"
        if 9 <= current_time.hour < 17 and random.random() < 0.05: # 5% chance of authorized RFID during 9 AM - 5 PM
            rfid_tag = random.choice(authorized_rfids)

        data.append([current_time, temp, gyro_x, gyro_y, gyro_z, rfid_tag, 0, "Normal"])

    # --- Generate Anomalous Data ---
    print(f"Generating {num_anomaly_entries} anomaly entries...")
    for _ in range(num_anomaly_entries):
        anomaly_type = random.choice([
            "Temperature Spike",
            "Temperature Drop",
            "Vault Movement (Minor)",
            "Vault Movement (Major)",
            "Unauthorized RFID",
            "RFID Jamming/No Read",
            "RFID Spamming"
        ])

        # Pick a random point in time near an existing normal entry
        # This makes anomalies appear in a realistic sequence
        random_normal_entry = random.choice(data)
        anomaly_time = random_normal_entry[0] + timedelta(seconds=random.randint(-30, 30)) # Offset slightly from a normal entry

        # Base normal values for the anomaly
        base_temp = random_normal_entry[1]
        base_gyro_x = random_normal_entry[2]
        base_gyro_y = random_normal_entry[3]
        base_gyro_z = random_normal_entry[4]
        base_rfid = random_normal_entry[5]

        # Initialize anomaly values with base values
        temp = base_temp
        gyro_x, gyro_y, gyro_z = base_gyro_x, base_gyro_y, base_gyro_z
        rfid_tag = base_rfid

        if anomaly_type == "Temperature Spike":
            temp = round(random.uniform(50.0, 100.0), 2) # High temperature due to cutting/drilling
        elif anomaly_type == "Temperature Drop":
            temp = round(random.uniform(-10.0, 5.0), 2) # Low temperature (e.g., using cryo-agents)
        elif anomaly_type == "Vault Movement (Minor)":
            gyro_x = round(random.uniform(-0.5, 0.5), 3)
            gyro_y = round(random.uniform(-0.5, 0.5), 3)
            gyro_z = round(random.uniform(-0.5, 0.5), 3)
        elif anomaly_type == "Vault Movement (Major)":
            gyro_x = round(random.uniform(-5.0, 5.0), 3)
            gyro_y = round(random.uniform(-5.0, 5.0), 3)
            gyro_z = round(random.uniform(-5.0, 5.0), 3)
        elif anomaly_type == "Unauthorized RFID":
            rfid_tag = random.choice(unauthorized_rfids)
        elif anomaly_type == "RFID Jamming/No Read":
            rfid_tag = "NONE" # Even if an authorized read was expected, it's jammed
        elif anomaly_type == "RFID Spamming":
            # Simulate multiple unauthorized reads in a short period
            # For this single entry, we'll just put an unauthorized tag
            # The model will need to learn patterns over time for this
            rfid_tag = random.choice(unauthorized_rfids) # Or even authorized tags at odd times/frequency

        data.append([anomaly_time, temp, gyro_x, gyro_y, gyro_z, rfid_tag, 1, anomaly_type])

    # Create DataFrame
    columns = ["timestamp", "temperature", "gyro_x", "gyro_y", "gyro_z", "rfid_tag_id", "is_anomaly", "anomaly_type"]
    df = pd.DataFrame(data, columns=columns)

    # Sort by timestamp to ensure chronological order
    df = df.sort_values(by="timestamp").reset_index(drop=True)

    # Save to CSV
    df.to_csv(output_filename, index=False)
    print(f"Dataset saved to {output_filename}")
    print(f"Total entries: {len(df)}")
    print(f"Anomaly ratio: {df['is_anomaly'].sum() / len(df):.2f}")
    return df

# --- Generate the dataset ---
if __name__ == "__main__":
    generated_df = generate_vault_hids_dataset(
        num_normal_entries=50000,   # You can increase this for more normal data
        num_anomaly_entries=5000    # And this for more anomalies
    )
    print("\nFirst 5 rows of the generated dataset:")
    print(generated_df.head())
    print("\nLast 5 rows of the generated dataset:")
    print(generated_df.tail())
    print("\nAnomaly distribution:")
    print(generated_df['is_anomaly'].value_counts())
    print("\nAnomaly types distribution:")
    print(generated_df[generated_df['is_anomaly'] == 1]['anomaly_type'].value_counts())