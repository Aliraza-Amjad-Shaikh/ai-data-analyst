import streamlit as st
from data_loader import load_and_profile_csv

# Simple test without running the full Streamlit app
if __name__ == "__main__":
    # Simulate a file upload by pointing to a local CSV file
    test_file_path = "test.csv"  # <<< CHANGE THIS TO A REAL CSV FILE ON YOUR PC
    
    try:
        # We use Streamlit's `open` to simulate an uploaded file object
        with open(test_file_path, "rb") as f:
            # Create a mock uploaded file object
            class MockUploadedFile:
                def __init__(self, file_path):
                    self.name = file_path
                    with open(file_path, 'rb') as file:
                        self.value = file.read()
                def getvalue(self):
                    return self.value
                    
            uploaded_file = MockUploadedFile(test_file_path)
            
            # Test our function
            df, profile, metadata = load_and_profile_csv(uploaded_file)
            
            print("✅ DataFrame loaded successfully!")
            print(f"📊 Shape: {df.shape}")
            print("\n--- GENERATED PROFILE ---\n")
            print(profile)
            print("\n--- METADATA ---\n")
            print(metadata)
            
    except FileNotFoundError:
        print(f"❌ Error: Please create a simple 'test.csv' file in your project folder first.")
        print("   You can create one easily in Excel or with this Python code:")
        print("   import pandas as pd; pd.DataFrame({'A': [1,2,3], 'B': ['x', 'y', 'z']}).to_csv('test.csv', index=False)")
    except Exception as e:
        print(f"❌ An error occurred: {e}")