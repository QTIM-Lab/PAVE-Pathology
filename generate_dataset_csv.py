import pandas as pd
from pathlib import Path
import os

def save_table(df, filename, title):
   os.makedirs("preprocessing", exist_ok=True)
   filepath = os.path.join("preprocessing", filename)
   
   if isinstance(df, pd.Series):
      df = df.to_frame()
   
   # Save as CSV
   df.to_csv(filepath)
   print(f"Saved table to {filepath}")
   
   # Also print nicely to console for immediate viewing
   print(f"\n--- {title} ---")
   print(df)

def main():
   print("--- Starting Dataset Generation ---")

   # 1. Load Data
   print("Loading labels.csv...")
   if not os.path.exists("labels.csv"):
      print("Error: labels.csv not found.")
      return
   
   master = pd.read_csv("labels.csv")
   initial_count = len(master)
   print(f"Initial rows: {initial_count}")

   # 2. Rename Columns
   master = master.rename(columns={
      'PATH_SERVER_NAME': 'country',
      'NCI_PID': 'patient_id',
      'ENRL_AGE': 'age',
      'PATH_MASKED_IMG_ID': 'image_id',
      'wst_bx_site': 'site_label',
      'path_expert_img': 'expert_label',
      'PATH_EXPERT_DX_OTHER': 'expert_label_other',
      'PATH_EXPERT_DX_NOTES': 'expert_label_notes',
      'PATH_LABEL_BIOPSY_TYPE': 'biopsy_type',
      'SCRN_HPV': 'hpv_type'
   })[[
      'image_id',
      'site_label',
      'expert_label',
      'expert_label_other',
      'expert_label_notes',
      'country',
      'patient_id',
      'age',
      'hpv_type',
      'biopsy_type',
      'svs_delivered',
   ]]

   # 3. Preprocessing
   master['image_id'] = master['image_id'].str.replace('.SVS', '')
   master['usable'] = 1
   master['label'] = master['expert_label']
   
   # 4. Label Mapping
   print("\nMapping labels...")
   master.loc[master.expert_label == "Insufficient/Inadequate", 'label'] = "insufficient"
   master.loc[master.expert_label == "Atypia: Specify", 'label'] = 'atypia'
   master.loc[master.expert_label == "Other: Specify", 'label'] = 'other'
   master.loc[master.expert_label == 'CIN1', 'label'] = 'low_grade'
   master.loc[master.expert_label.isin(['CIN2','CIN3','AIS']), 'label'] = 'high_grade'
   
   cancer_condition = (
      master.expert_label.isin([
         'Adenocarcinoma Invasive','Adenosquamous Carcinoma','Other Cancer: Specify','Squamous Invasive Carcinoma'
      ]) | 
      (
         (master.expert_label == "Other: Specify") & 
         (master.expert_label_other.str.contains("carcinoma", case=False, na=False)) &
         (~master.expert_label_other.str.contains("rule out", case=False, na=False))
      )
   )
   master.loc[cancer_condition, 'label'] = 'cancer'

   normal_condition = (
      (master.expert_label == "Negative/Reactive") |
      (
         (master.expert_label == "Other: Specify") &
         (master.expert_label_other.str.contains("microglandular hyperplasia", case=False, na=False))
      )
   )
   master.loc[normal_condition, 'label'] = 'normal'

   # 5. Define Exclusion/Inclusion Masks
   print("\n--- Defining Exclusion/Inclusion Masks ---")

   # Mask 1: Unreviewed
   unreviewed = master.expert_label == " "
   
   # Mask 2: Malawi Normals
   malawi_neg = (master['country'] == "Malawi") & (master['label'] == "normal")
   
   # Mask 3: Other/Atypia
   other_atypia = (master.label == "other") | (master.label == "atypia")
   
   # Mask 4: LEEP
   # string_cols = master.select_dtypes(include=['object']).columns
   # leeps = master[string_cols].apply(lambda s: s.str.contains("LEEP", case=False, na=False)).any(axis=1)
   
   # Mask 5: Not Delivered
   not_delivered = (master.svs_delivered != 1)

   # Mask 6: Not Ready (Missing .pt files)
   print("Checking for processed .pt files in /scratch90/taghinia/pave_training/uni_v1 ...")
   base = Path("/scratch90/taghinia/pave_training/uni_v1")
   found_ids = set()
   for f in base.rglob("pt_files/*.pt"):
      image_id = f.parts[-1].split('.')[0]
      found_ids.add(image_id)
   
   not_ready = ~master.image_id.isin(found_ids)

   # Combine masks into a DataFrame for analysis
   masks = pd.DataFrame({
       'Unreviewed': unreviewed,
       'Malawi_Neg': malawi_neg,
       'Other_Atypia': other_atypia,
       # 'LEEP': leeps,
       'Not_Delivered': not_delivered,
       'Not_Ready': not_ready
   })

   # Calculate 'Any Exclusion'
   masks['Any_Exclusion'] = masks.any(axis=1)

   # --- 1. Inclusion Flow Table (Sequential) ---
   print("\n--- Generating Inclusion Flow Table ---")
   steps = [
       ('Unreviewed', unreviewed),
       ('Malawi Negative/Reactive', malawi_neg),
       ('Other/Atypia', other_atypia),
       # ('LEEP', leeps),
       ('Not Delivered', not_delivered),
       ('Not Processed (Missing .pt)', not_ready)
   ]
   
   flow_data = []
   current_mask = pd.Series(True, index=master.index)
   n_slides_start = len(master)
   n_patients_start = master['patient_id'].nunique()
   
   flow_data.append({
       'Step': 'Initial Dataset',
       'Excluded Slides': 0,
       'Excluded Patients (Dropped)': 0,
       'Remaining Slides': n_slides_start,
       'Remaining Patients': n_patients_start
   })
   
   prev_n_patients = n_patients_start

   for step_name, step_mask in steps:
       # Slides to exclude in this step (must be currently valid AND match exclusion criteria)
       slides_to_exclude = current_mask & step_mask
       n_excluded_slides = slides_to_exclude.sum()
       
       # Update the current mask (keep only those NOT excluded)
       current_mask = current_mask & (~step_mask)
       
       n_remaining_slides = current_mask.sum()
       n_remaining_patients = master.loc[current_mask, 'patient_id'].nunique()
       n_dropped_patients = prev_n_patients - n_remaining_patients
       
       flow_data.append({
           'Step': f"Exclude {step_name}",
           'Excluded Slides': n_excluded_slides,
           'Excluded Patients (Dropped)': n_dropped_patients,
           'Remaining Slides': n_remaining_slides,
           'Remaining Patients': n_remaining_patients
       })
       prev_n_patients = n_remaining_patients

   flow_df = pd.DataFrame(flow_data)
   save_table(flow_df, "inclusion_flow.csv", "Data Inclusion Flow")

   # --- 2. Co-occurrence Matrix (Parallel) ---
   print("\n--- Detailed Overlap (Co-occurrence Matrix) ---")
   exclusion_only = masks.drop(columns=['Any_Exclusion'])
   cooccurrence = exclusion_only.astype(int).T.dot(exclusion_only.astype(int))
   save_table(cooccurrence, "exclusion_overlap_matrix.csv", "Exclusion Criteria Co-occurrence Matrix")

   # 6. Apply Filters
   print("\n--- Applying Filters ---")
   # We keep rows where 'Any_Exclusion' is False
   clean_mask = ~masks['Any_Exclusion']
   prepped = master[clean_mask].copy()

   # 7. Final Dataset Creation
   prepped = prepped[['patient_id', 'image_id', 'label', 'country', 'age']].rename(
      columns={
         'patient_id': 'case_id',
         'image_id': 'slide_id'
      }
   )

   # 8. Save and Print Stats
   print("\n--- Final Dataset Stats ---")
   print(f"N.patients: {prepped.case_id.nunique()}")
   print(f"N.slides: {len(prepped)}")
   
   output_path = "dataset_csv/with_leeps.csv"
   os.makedirs(os.path.dirname(output_path), exist_ok=True)
   prepped.to_csv(output_path)
   print(f"Saved to {output_path}")

   print("Saving stats tables to preprocessing/ folder...")
   
   age_stats = prepped['age'].astype(float).describe()
   save_table(age_stats, "age_stats.csv", "Descriptive Stats for Age")

   label_counts = prepped.label.value_counts()
   save_table(label_counts, "label_distribution.csv", "Label Distribution")
   
   ct = pd.crosstab(prepped.country, prepped.label)
   ct['Total'] = ct.sum(axis=1)
   # Add total row as well
   ct.loc['Total'] = ct.sum()
   save_table(ct, "country_label_crosstab.csv", "Crosstab: Country vs Label")

if __name__ == "__main__":
   main()
