import pandas as pd
from datetime import datetime

class Logger:
    def __init__(self):
        self.log_data = []

    def log_motion_event(self):
        self.log_data.append({
            'timestamp': datetime.now(),
            'event_type': 'motion_detected'
        })

    def save_logs(self, filename='motion_logs.csv'):
        df = pd.DataFrame(self.log_data)
        df.to_csv(filename, index=False)