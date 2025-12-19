"""
Quick script to clear all cached LSTM+GRU models
Run this after fixing the drift issue to force retraining with new constraints
"""
import os
import shutil

def clear_model_cache():
    """Clear all cached models"""
    cache_dirs = [
        'trained_models',
        'model_cache',
        'mcp_tools/trained_models'
    ]
    
    total_deleted = 0
    
    for cache_dir in cache_dirs:
        if os.path.exists(cache_dir):
            print(f"\nClearing {cache_dir}...")
            try:
                for file in os.listdir(cache_dir):
                    file_path = os.path.join(cache_dir, file)
                    try:
                        if os.path.isfile(file_path):
                            os.unlink(file_path)
                            print(f"  Deleted: {file}")
                            total_deleted += 1
                    except Exception as e:
                        print(f"  Error deleting {file}: {e}")
            except Exception as e:
                print(f"  Error accessing {cache_dir}: {e}")
        else:
            print(f"  {cache_dir} does not exist")
    
    print(f"\nModel cache cleared! Deleted {total_deleted} cached models.")
    print("Models will retrain automatically on next prediction request.")

if __name__ == "__main__":
    clear_model_cache()
