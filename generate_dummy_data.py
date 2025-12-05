import csv
import random

def create_employees_csv(filename="employees.csv"):
    headers = ["Name", "Department", "Age", "Salary", "Years_Experience"]
    data = [
        ["Alice", "Engineering", 30, 120000, 5],
        ["Bob", "HR", 45, 75000, 15],
        ["Charlie", "Sales", 28, 50000, 2],
        ["David", "Engineering", 35, 135000, 10],
        ["Eve", "Marketing", 32, 90000, 8],
        ["Frank", "Sales", 40, 85000, 12],
        ["Grace", "HR", 29, 60000, 3],
        ["Hannah", "Engineering", 26, 95000, 2],
        ["Ivan", "Marketing", 50, 110000, 20],
        ["Judy", "Sales", 24, 45000, 1]
    ]
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(data)
    print(f"Created {filename}")

def create_products_csv(filename="products.csv"):
    headers = ["Product_ID", "Product_Name", "Category", "Price", "Stock_Quantity"]
    data = [
        [101, "Laptop", "Electronics", 999.99, 50],
        [102, "Mouse", "Electronics", 25.50, 200],
        [103, "Desk Chair", "Furniture", 150.00, 15],
        [104, "Coffee Table", "Furniture", 89.99, 10],
        [105, "Headphones", "Electronics", 199.99, 75],
        [106, "Notebook", "Stationery", 5.00, 500],
        [107, "Pen Set", "Stationery", 12.99, 100],
        [108, "Monitor", "Electronics", 250.00, 40],
        [109, "Bookshelf", "Furniture", 120.00, 20],
        [110, "Stapler", "Stationery", 7.50, 150]
    ]
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(data)
    print(f"Created {filename}")

def create_students_csv(filename="students.csv"):
    headers = ["Student_ID", "Name", "Math_Score", "Science_Score", "English_Score"]
    names = ["Liam", "Olivia", "Noah", "Emma", "Oliver", "Ava", "Elijah", "Charlotte", "William", "Sophia"]
    data = []
    for i, name in enumerate(names, 1):
        data.append([
            i, 
            name, 
            random.randint(60, 100), # Math
            random.randint(60, 100), # Science
            random.randint(60, 100)  # English
        ])
    
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(data)
    print(f"Created {filename}")

if __name__ == "__main__":
    create_employees_csv()
    create_products_csv()
    create_students_csv()
    print("\nAll files generated successfully! You can now use 'employees.csv', 'products.csv', or 'students.csv' in your agent.")