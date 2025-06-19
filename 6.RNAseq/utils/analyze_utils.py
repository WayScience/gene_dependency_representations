import pandas as pd    


def find_drugs(predictions_df, r2_df, drug_df):
    # Filter the test results for R² score over 0.6
    filtered_r2_df = r2_df[r2_df['R2_score'] > 0.2]

    pred_df = predictions_df

    # Initialize a results dictionary to store the top correlations for each case
    results = {}

    # Loop through each row in the filtered R² DataFrame
    for _, row in filtered_r2_df.iterrows():
        latent_dim = row['latent_dimension']
        z_dim = row['z_dimension']  # Use extracted numeric z
        z_numeric = int(z_dim.split('_')[1])  # Extract numeric part of 'z_' (e.g., 'z_5' -> 5)
        model = row['model']
        init = row['init']
        
        # Filter the correlation DataFrame based on these conditions
        filtered_df = drug_df[
            (drug_df['full_model_z'] == latent_dim) &
            (drug_df['z'] == z_numeric) &
            (drug_df['model'] == model) &
            (drug_df['init'] == init)
        ]

        # Select the row with the highest absolute Pearson correlation
        if not filtered_df.empty:
            top_drug_row = filtered_df.loc[filtered_df['pearson_correlation'].abs().idxmax()]
            
            # Add a new row with type 'drug' and the chosen drug
            new_drug_row = pd.DataFrame([{
                'model': model,
                'latent_dimension': latent_dim,
                'z_dimension': f'z_{z_numeric}',  # Ensure 'z_' format
                'type': 'drug',
                'drug': top_drug_row['drug'],
                'pearson_correlation': top_drug_row['pearson_correlation'],
                'init': init,
            }])

            # Append the new row to the dataframe
            pred_df = pd.concat([pred_df, new_drug_row], ignore_index=True)

            # For the predicted and actual types, set 'drug' column to n/a
            pred_df.loc[
                (pred_df['model'] == model) &
                (pred_df['latent_dimension'] == latent_dim) &
                (pred_df['z_dimension'] == f'z_{z_numeric}') &
                (pred_df['init'] == init) &
                (pred_df['type'].isin(['predicted', 'actual'])),
                'drug'
            ] = 'n/a'

            # Store the result in the dictionary for verification
            results[(latent_dim, z_dim, model, init)] = top_drug_row

        # Check and print results
        for key, top_drug_row in results.items():
            latent_dim, z_dim, model, init = key
            print(f"Top drug for latent_dimension: {latent_dim}, z_dimension: {z_dim}, model: {model}, init: {init}")
            print(top_drug_row)
            print()
    return pred_df