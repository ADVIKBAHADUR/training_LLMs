import os
import pandas as pd
from tensorboard.backend.event_processing import event_accumulator
from pathlib import Path

def extract_tensorboard_losses(log_dir, output_csv='tensorboard_losses.csv'):
    """
    Extract Loss data from TensorBoard logs.
    
    Args:
        log_dir: Path to the directory containing TensorBoard event files
        output_csv: Output CSV filename
    """
    
    # Dictionary to store all data: {step: {model_train: value, model_val: value}}
    all_data = {}
    
    # Find all event files recursively
    for root, dirs, files in os.walk(log_dir):
        for file in files:
            if file.startswith('events.out.tfevents'):
                event_file_path = os.path.join(root, file)
                
                # Get the run name (relative path from log_dir)
                run_path = os.path.relpath(root, log_dir)
                
                try:
                    # Load the event file
                    ea = event_accumulator.EventAccumulator(event_file_path)
                    ea.Reload()
                    
                    # Get all scalar tags
                    tags = ea.Tags()['scalars']
                    
                    # Process all tags
                    for tag in tags:
                        # Only process 'Loss' tag
                        if tag != 'Loss':
                            continue
                        
                        # Determine model name and metric type from the directory path
                        path_parts = run_path.split(os.sep)
                        
                        # Extract model name and metric type
                        model_name = None
                        metric_type = None
                        
                        for part in path_parts:
                            if 'Loss_Train' in part:
                                model_name = path_parts[0]  # First part is model name
                                metric_type = 'Train'
                                break
                            elif 'Loss_Val' in part:
                                model_name = path_parts[0]  # First part is model name
                                metric_type = 'Val'
                                break
                        
                        # Skip if we couldn't determine the model/metric
                        if not model_name or not metric_type:
                            continue
                        
                        column_name = f"{model_name}_{metric_type}"
                        
                        # Extract scalar events
                        scalar_events = ea.Scalars(tag)
                        
                        # Store data
                        for event in scalar_events:
                            step = event.step
                            value = event.value
                            
                            if step not in all_data:
                                all_data[step] = {}
                            
                            all_data[step][column_name] = value
                    
                    print(f"Processed: {run_path} - Found tags: {tags}")
                    
                except Exception as e:
                    print(f"Error processing {event_file_path}: {e}")
    
    # Convert to DataFrame
    if not all_data:
        print("No data found. Make sure the log directory path is correct.")
        return None
    
    df = pd.DataFrame.from_dict(all_data, orient='index')
    df.index.name = 'Step'
    df = df.sort_index()
    
    # Reorder columns: alternating Train/Val for each model
    columns = sorted(df.columns)
    train_cols = [c for c in columns if c.endswith('_Train')]
    val_cols = [c for c in columns if c.endswith('_Val')]
    
    # Group by model
    models = set([c.rsplit('_', 1)[0] for c in columns])
    ordered_columns = []
    for model in sorted(models):
        if f"{model}_Train" in train_cols:
            ordered_columns.append(f"{model}_Train")
        if f"{model}_Val" in val_cols:
            ordered_columns.append(f"{model}_Val")
    
    df = df[ordered_columns]
    
    # Save to CSV
    df.to_csv(output_csv)
    print(f"\nData extracted successfully!")
    print(f"Shape: {df.shape}")
    print(f"Saved to: {output_csv}")
    print(f"\nColumns: {list(df.columns)}")
    print(f"\nFirst few rows:")
    print(df.head())
    print(f"\nLast few rows:")
    print(df.tail())
    
    return df

# Usage
if __name__ == "__main__":
    # Replace with your actual TensorBoard log directory
    LOG_DIRECTORY = './runs/selected_models'
    OUTPUT_FILE = 'tensorboard_losses.csv'
    
    print("TensorBoard Loss Extraction Script")
    print("=" * 50)
    print(f"Log directory: {LOG_DIRECTORY}")
    print(f"Output file: {OUTPUT_FILE}")
    print("=" * 50)
    
    df = extract_tensorboard_losses(LOG_DIRECTORY, OUTPUT_FILE)
