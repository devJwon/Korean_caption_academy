import re
import pandas as pd

# Extract the number from the response_text column and create a new column to check if the selected caption matches the real_caption

# Function to extract number from response_text
def extract_number_from_string(input_string):

    # Simplified pattern to match any digit(s) within the string, preferring those within curly braces if present
    matches = re.findall(r'\d+', input_string)
    
    if matches:
        return matches[0]  # Return the first number found
    else:
        return None

    
def calculate_accuracy(df):
    # Calculate the number of correct predictions
    correct_predictions = df['correct'].sum()
    # Calculate the total number of predictions
    total_predictions = len(df)
    # Calculate and return the accuracy
    accuracy = (correct_predictions / total_predictions) * 100
    return accuracy

data = pd.read_csv('result/claude_Ko_response_texts.csv')
# Apply the function to extract numbers
data['selected_caption_number'] = data['response_text'].apply(extract_number_from_string)

# Function to check if selected caption matches the real_caption
def is_correct(row):
    print(row['selected_caption_number'])
    if row['selected_caption_number'] :
        # Column name for the selected caption based on the extracted number
        selected_caption_column = f'caption{row["selected_caption_number"]}'
    # Check if the selected caption matches the real_caption
        return row[selected_caption_column] == row['real_caption']
    else:
        return False
# Apply the function to create the 'correct' column
data['correct'] = data.apply(is_correct, axis=1)

# Display the modified dataframe with the new 'correct' column
print(calculate_accuracy(data))