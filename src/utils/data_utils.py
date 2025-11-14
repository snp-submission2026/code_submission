import csv
import re

def get_data_from_text_file(file_path):
    data = []
    current_tokens = []
    meta_id = None
    context_label = None
    hindi_count = 0
    english_count = 0
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split()

            # Detect meta line: e.g., ['meta', '1', 'positive']
            if parts[0] == "meta" and len(parts) == 3:
                # Save previous block before starting a new one
                if meta_id is not None and current_tokens:
                    text = " ".join(current_tokens).strip()
                    lang = "hi" if hindi_count >= english_count else "en"
                    data.append([meta_id, text, context_label, lang])
                    current_tokens = []
                    hindi_count = 0
                    english_count = 0

                meta_id = parts[1]
                context_label = parts[2]
                continue

            # Handle token lines (e.g., ['So', 'lang1'])
            if len(parts) >= 1:
                try:
                    token = parts[0]
                    current_tokens.append(token)
                    hindi_count += parts[1] == "Hin"
                    english_count += parts[1] == "Eng"
                except IndexError:
                    # If language tag is missing, just add the token
                    current_tokens.append(parts[0])

    # Save the last block
    if meta_id is not None and current_tokens:
        text = " ".join(current_tokens).strip()
        lang = "hi" if hindi_count >= english_count else "en"
        data.append([meta_id, text, context_label, lang])

    return data
