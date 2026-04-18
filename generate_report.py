import pandas as pd
import numpy as np
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
import os

def generate_report():
    """Generates a data drift report using Evidently AI."""
    print("Generating Drift Report...")
    
    # Simulate reference and current data
    np.random.seed(42)
    reference_data = pd.DataFrame({
        'Avg_min_between_sent_tnx': np.random.uniform(0, 1000, 500),
        'Avg_min_between_received_tnx': np.random.uniform(0, 1000, 500),
        'Time_Diff_between_first_and_last_Mins': np.random.uniform(0, 10000, 500),
        'Sent_tnx': np.random.randint(0, 100, 500),
        'Received_Tnx': np.random.randint(0, 100, 500),
        'Number_of_Created_Contracts': np.random.randint(0, 10, 500),
        'target': np.random.randint(0, 2, 500)
    })
    
    # Introduce drift in current data
    current_data = reference_data.copy()
    current_data['Sent_tnx'] = current_data['Sent_tnx'] * 1.5 
    
    # Create Report
    report = Report(metrics=[
        DataDriftPreset(),
        TargetDriftPreset(),
    ])
    
    report.run(reference_data=reference_data, current_data=current_data)
    
    # Save as HTML
    output_path = "drift_report.html"
    report.save_html(output_path)
    print(f"Report saved to {output_path}")

if __name__ == "__main__":
    # Ensure evidently is installed for the demo to work
    try:
        generate_report()
    except ImportError:
        print("Evidently AI not installed. Run 'pip install evidently' first.")
