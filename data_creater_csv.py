import pandas as pd
sample_data = {
    'Product': ['Widget A', 'Widget B', 'Widget C'],
    'Price': [19.99, 24.99, 15.50],
    'Units_Sold': [100, 85, 120],
    'Customer_Rating': [4.5, 4.2, 4.8]
}
df = pd.DataFrame(sample_data)
df.to_csv('test.csv', index=False)
print("test.csv created!")