#!/usr/bin/env python3
"""
NeuroGait File Path Checker
===========================

Quick script to verify the location of your Excel data file.
"""

import os
import pandas as pd

def check_file_paths():
    """Check various possible locations for the Excel file"""
    
    print("🔍 NeuroGait File Path Checker")
    print("=" * 40)
    
    # Possible file paths to check
    possible_paths = [
        "Final dataset.xlsx",
        "GiorgosBouh/NeuroGait_ASD/Final dataset.xlsx", 
        os.path.expanduser("~/NeuroGait_ASD/Final dataset.xlsx"),
        os.path.expanduser("~/GiorgosBouh/NeuroGait_ASD/Final dataset.xlsx"),
        "../Final dataset.xlsx",
        "./Final dataset.xlsx"
    ]
    
    print(f"📁 Current directory: {os.getcwd()}")
    print(f"🏠 Home directory: {os.path.expanduser('~')}")
    print()
    
    print("Checking possible file locations:")
    print("-" * 40)
    
    found_files = []
    
    for i, path in enumerate(possible_paths, 1):
        exists = os.path.exists(path)
        status = "✅ FOUND" if exists else "❌ Not found"
        print(f"{i}. {path}")
        print(f"   {status}")
        
        if exists:
            try:
                # Try to read the file to verify it's valid
                df = pd.read_excel(path)
                print(f"   📊 File info: {len(df)} rows, {len(df.columns)} columns")
                
                # Check for expected columns
                has_class = 'class' in df.columns
                class_status = "✅ Has 'class' column" if has_class else "⚠️  Missing 'class' column"
                print(f"   {class_status}")
                
                if has_class:
                    unique_classes = df['class'].unique()
                    print(f"   🏷️  Classifications: {list(unique_classes)}")
                
                found_files.append(path)
                
            except Exception as e:
                print(f"   ❌ Error reading file: {e}")
        
        print()
    
    print("=" * 40)
    
    if found_files:
        print(f"✅ Found {len(found_files)} valid file(s):")
        for file_path in found_files:
            print(f"   📄 {file_path}")
        
        print()
        print("💡 Recommended action:")
        print(f"   Use this path in your script: '{found_files[0]}'")
        
        # Show how to update the script
        print()
        print("🔧 To update the knowledge graph builder:")
        print(f"   Edit neurogait_kg_builder.py")
        print(f"   Change EXCEL_FILE = \"{found_files[0]}\"")
    
    else:
        print("❌ No valid Excel files found!")
        print()
        print("💡 Troubleshooting steps:")
        print("1. Verify the file exists and is named 'Final dataset.xlsx'")
        print("2. Check if you're in the correct directory")
        print("3. Ensure the file is not corrupted")
        print("4. Try using the absolute path to the file")
    
    print()
    print("🗂️  Current directory contents:")
    try:
        files = [f for f in os.listdir('.') if f.endswith(('.xlsx', '.xls', '.csv'))]
        if files:
            for file in files:
                print(f"   📄 {file}")
        else:
            print("   (No Excel/CSV files found)")
    except Exception as e:
        print(f"   ❌ Error listing directory: {e}")


if __name__ == "__main__":
    check_file_paths()