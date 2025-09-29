import pandas as pd
import json

def create_dataset_json():
    # Read the CSV file
    df = pd.read_csv('train.csv')
    
    # Filter only valid entries
    df = df[df['valid'] == 'Yes']
    
    # Create the dataset list
    dataset = []
    
    # Define the question based on the provided template
    question = (
        "Transcription: \"{transcription}\"\n\n"
        "Based on the speech audio and its transcription, "
        "classify the speaker's cognitive status with a single word "
        "from the following options: 'ad' (Alzheimer's Disease), "
        "'mci' (Mild Cognitive Impairment), or 'cn' (Cognitively Normal)."
    )
    
    for _, row in df.iterrows():
        uid = row['uid']
        transcription = row['transcription']
        
        # Determine the answer based on diagnosis columns
        if row['diagnosis_adrd'] == 1.0:
            answer = 'ad'
        elif row['diagnosis_mci'] == 1.0:
            answer = 'mci'
        elif row['diagnosis_control'] == 1.0:
            answer = 'cn'
        else:
            # Skip rows where diagnosis is unclear
            continue
        
        # Format the question with the transcription
        formatted_question = question.format(transcription=transcription)
        
        # Create the entry following the mllm_audio_demo.json format
        entry = {
            "messages": [
                {
                    "content": f"<audio>{formatted_question}",
                    "role": "user"
                },
                {
                    "content": answer,
                    "role": "assistant"
                }
            ],
            "audios": [
                f"data/denoised_data/low-pass_filter/LPF_train_audios/{uid}.wav"
            ]
        }
        
        dataset.append(entry)
    
    # Write to JSON file
    with open('cognitive_status_dataset.json', 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
    
    print(f"Created dataset with {len(dataset)} entries")
    print(f"Saved to: cognitive_status_dataset.json")
    
    # Print some statistics
    answers = [entry['messages'][1]['content'] for entry in dataset]
    print(f"\nDataset statistics:")
    print(f"Total entries: {len(answers)}")
    print(f"CN (Cognitively Normal): {answers.count('cn')}")
    print(f"MCI (Mild Cognitive Impairment): {answers.count('mci')}")
    print(f"AD (Alzheimer's Disease): {answers.count('ad')}")

if __name__ == "__main__":
    create_dataset_json() 