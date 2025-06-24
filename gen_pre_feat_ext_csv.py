import pandas as pd
import os
def process_csv_for_feature_extraction(csv_path):
   """
   Process the CSV file for feature extraction by removing the .svs file extension from the 'slide_id' column
   and filtering out rows where 'status' is not 'processed' or 'already_exist'.

   Args:
      csv_path (str): Path to the input CSV file.

   Returns:
      pd.DataFrame: Processed DataFrame ready for feature extraction.
   """
   # Load the CSV file into a DataFrame
   try:
      df = pd.read_csv(csv_path)
      print(f"Loaded CSV with {len(df)} rows.")
   except Exception as e:
      print(f"Error loading {csv_path}: {e}")
      exit()

   # Remove the .svs file extension from the 'slide_id' column
   df['slide_id'] = df['slide_id'].str.replace('.svs', '', regex=False)
   print(f"Removed .svs extensions from slide IDs")

   # Filter out rows where 'status' is 'failed_seg'
   df = df[df['status'] != 'failed_seg']
   print(f"Filtered out slides that failed segmentation, remaining rows: {len(df)}")

   df['status'] = 'tbp'

   return df

def write_processed_csv(df, output_dir):
   """
   Write the processed DataFrame to a new CSV file.

   Args:
      df (pd.DataFrame): Processed DataFrame.
      output_path (str): Path to save the processed CSV file.
   """
   output_path = os.path.join(output_dir, 'pre_feat_ext.csv')
   print(f"Writing to {output_path}...")
   df.to_csv(output_path, index=False)
   print(f"Processed CSV saved to {output_path}")

if __name__ == "__main__":
   import argparse
   
   parser = argparse.ArgumentParser(description="Process auto-generated segmentation CSV into a CSV for feature extraction.")
   parser.add_argument('--input_csv', type=str, help='Path to the input CSV file.')
   parser.add_argument('--output_dir', type=str, help='Path to save the processed CSV file.')

   args = parser.parse_args()

   # Process the input CSV
   processed_df = process_csv_for_feature_extraction(args.input_csv)

   if processed_df is not None:
      # Write the processed DataFrame to a new CSV file
      write_processed_csv(processed_df, args.output_dir)
   else:
      print("No processed CSV file was created.")