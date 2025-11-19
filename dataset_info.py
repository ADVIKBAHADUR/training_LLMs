import os
import json
import matplotlib.pyplot as plt
from collections import Counter

def analyse_dataset(path):
    missing_chars = []
    important_vocabs = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")

    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    text_chars = set(text)
    missing_chars = sorted(important_vocabs - text_chars)

    if missing_chars:
        print(f"Missing {len(missing_chars)} characters:")
        print(missing_chars)
    else:
        print("All important characters are present.")


    chars = sorted(list(set(text)))
    counter = Counter(text)
    top_10 = counter.most_common(10)

    # Create a mapping for special characters
    label_mapping = {
        ' ': 'space',
        '\n': 'newline',
        '\t': 'tab',
        '\r': 'carriage return'
    }

    # Replace characters with readable labels
    characters = [label_mapping.get(char, char) for char, count in top_10]
    frequencies = [count for char, count in top_10]

    plt.bar(characters, frequencies)
    plt.xlabel('Characters')
    plt.ylabel('Frequency')
    plt.title('Top 10 Most Common Characters for Dataset: ' + os.path.basename(path))
    plt.xticks(rotation=45, ha='right')  # Rotate labels for readability
    plt.tight_layout()
    plt.show()
    num_chars = len(text)
    num_lines = text.count("\n") + 1
    vocab_size = len(chars)

    # Character-type analysis
    num_alpha = sum(c.isalpha() for c in text)
    num_digits = sum(c.isdigit() for c in text)
    num_spaces = text.count(" ")
    num_punct = sum(c in ".,'\"!?;:-()[]{}" for c in text)

    unusual_chars = [c for c in chars if not (c.isalpha() or c.isdigit() or c in " .,!?;'\"-()[]{}")]

    avg_line_length = num_chars / num_lines

    stats = {
        "dataset_path": path,
        "num_characters": num_chars,
        "num_lines": num_lines,
        "average_line_length": avg_line_length,
        "vocab_size": vocab_size,
        "vocabulary_preview": chars[:50],
        "first_300_characters": text[:300],
        "character_frequencies_top_20": counter.most_common(20),
        "num_alphabetic_characters": num_alpha,
        "num_digits": num_digits,
        "num_spaces": num_spaces,
        "num_punctuation_characters": num_punct,
        "unusual_characters": unusual_chars,
        "no_missing_important_characters": len(missing_chars),
        "missing_important_characters": missing_chars
    }

    return stats


# ---------------------------------------------------------
# Main function to process multiple datasets
# ---------------------------------------------------------

def inspect_datasets(dataset_paths, output_dir="dataset_reports"):
    os.makedirs(output_dir, exist_ok=True)

    summary = {}

    for path in dataset_paths:
        name = os.path.splitext(os.path.basename(path))[0]
        print(f"\n====================================")
        print(f" ANALYSING DATASET: {name}")
        print("====================================")

        stats = analyse_dataset(path)

        # Print a clean human-readable summary
        print(f"📄 File: {stats['dataset_path']}")
        print(f"➡️ Characters:   {stats['num_characters']}")
        print(f"➡️ Lines:        {stats['num_lines']}")
        print(f"➡️ Avg line len: {stats['average_line_length']:.2f}")
        print(f"➡️ Vocabulary:   {stats['vocab_size']} unique symbols")
        print(f"➡️ Sample preview:\n{stats['first_300_characters']}")
        print(f"➡️ Unusual characters: {stats['unusual_characters']}\n")
        print(f"➡️ Missing important characters: {stats['missing_important_characters']}\n")
        

        # Save JSON
        out = os.path.join(output_dir, name + "_stats.json")
        with open(out, "w") as f:
            json.dump(stats, f, indent=4)

        summary[name] = stats
        print(f"✓ Saved report to {out}")

    # Save global summary
    with open(os.path.join(output_dir, "ALL_DATASETS_SUMMARY.json"), "w") as f:
        json.dump(summary, f, indent=4)

    print("\n🎉 All dataset analyses complete!")
    return summary

def get_comparisons():
    with open("dataset_reports/ALL_DATASETS_SUMMARY.json", "r") as f:
        summary = json.load(f)

    dataset_names = list(summary.keys())
    vocab_sizes = [summary[name]["vocab_size"] for name in dataset_names]
    num_characters = [summary[name]["num_characters"] for name in dataset_names]
    avg_line_lengths = [summary[name]["average_line_length"] for name in dataset_names]
    x = range(len(dataset_names))

    fig, axs = plt.subplots(3, 1, figsize=(10, 15))
    fig.subplots_adjust(hspace=0.4)  # Increase hspace to add vertical space between subplots

    # Plot 1: Vocabulary Size
    axs[0].bar(x, vocab_sizes, color='skyblue')
    axs[0].set_xticks(x)
    axs[0].set_xticklabels(dataset_names, ha='center')
    axs[0].set_ylabel('Vocabulary Size')
    axs[0].set_title('Vocabulary Size across Datasets', loc='center', pad=0)

    # Plot 2: Number of Characters
    axs[1].bar(x, num_characters, color='lightgreen')
    axs[1].set_xticks(x)
    axs[1].set_xticklabels(dataset_names, ha='center')
    axs[1].set_ylabel('Number of Characters')
    axs[1].set_title('Number of Characters across Datasets', loc='center', pad=0)

    # Plot 3: Average Line Length
    axs[2].bar(x, avg_line_lengths, color='salmon')
    axs[2].set_xticks(x)
    axs[2].set_xticklabels(dataset_names, ha='center')
    axs[2].set_ylabel('Average Line Length')
    axs[2].set_title('Average Line Length across Datasets', loc='center', pad=0)

    plt.tight_layout(pad=3.0)  # Default is 1.08; increase for more spacing
    plt.show()

    

if __name__ == "__main__":
    DATASETS = [
        "datasets/input_childSpeech_trainingSet.txt",
        "datasets/input_childSpeech_testSet.txt",
        "datasets/input_shakespeare.txt"
    ]

    inspect_datasets(DATASETS)
    get_comparisons()
