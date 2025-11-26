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
        "character_frequencies_top_10": top_10,
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
    all_stats = {}

    # simple mapping to make common invisible characters readable on plots
    label_mapping = {
        ' ': 'space',
        '\n': 'newline',
        '\t': 'tab',
        '\r': 'carriage return'
    }

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
        with open(out, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=4, ensure_ascii=False)

        summary[name] = stats
        all_stats[name] = stats
        print(f"✓ Saved report to {out}")

    # Create plots
    # If both child speech training and test are present, create a combined grouped bar chart
    child_train_key = os.path.splitext(os.path.basename(dataset_paths[0]))[0]  # placeholder init
    # find correct keys by name matching
    keys = list(all_stats.keys())
    child_train_key = next((k for k in keys if "childSpeech_trainingSet" in k or "input_childSpeech_trainingSet" in k), None)
    child_test_key = next((k for k in keys if "childSpeech_testSet" in k or "input_childSpeech_testSet" in k), None)

    # Helper to convert characters to readable labels
    def readable(chars):
        return [label_mapping.get(c, c) for c in chars]

    if child_train_key and child_test_key:
        s1 = all_stats[child_train_key]['character_frequencies_top_10']
        s2 = all_stats[child_test_key]['character_frequencies_top_10']

        chars1 = [c for c, _ in s1]
        freqs1 = [cnt for _, cnt in s1]
        # align s2 to chars1 order (if some char missing in s2, use 0)
        freq_map2 = {c: cnt for c, cnt in s2}
        freqs2 = [freq_map2.get(c, 0) for c in chars1]

        x = range(len(chars1))
        width = 0.35

        plt.figure(figsize=(12, 6))
        plt.bar([i - width/2 for i in x], freqs1, width=width, label=child_train_key, color='skyblue')
        plt.bar([i + width/2 for i in x], freqs2, width=width, label=child_test_key, color='salmon')
        plt.xlabel('Characters')
        plt.ylabel('Frequency')
        plt.title('Top 10 Character Frequencies: Child Speech (Train vs Test)')
        plt.xticks(x, readable(chars1), rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        out_plot = os.path.join(output_dir, f"{child_train_key}_vs_{child_test_key}_top10.png")
        plt.savefig(out_plot, dpi=150)
        plt.show()
        print(f"✓ Saved combined child-speech comparison plot to {out_plot}")

        # For other datasets, plot individually
        for name, stats in all_stats.items():
            if name in (child_train_key, child_test_key):
                continue
            top10 = stats['character_frequencies_top_10']
            chars = [c for c, _ in top10]
            freqs = [cnt for _, cnt in top10]

            plt.figure(figsize=(8, 4))
            plt.bar(readable(chars), freqs, color='lightgreen')
            plt.xlabel('Characters')
            plt.ylabel('Frequency')
            plt.title(f"Top 10 Most Common Characters for Dataset: {name}")
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            out_plot = os.path.join(output_dir, f"{name}_top10.png")
            plt.savefig(out_plot, dpi=150)
            plt.show()
            print(f"✓ Saved plot to {out_plot}")
    else:
        # fallback: plot each dataset individually as before
        for name, stats in all_stats.items():
            top10 = stats['character_frequencies_top_10']
            chars = [c for c, _ in top10]
            freqs = [cnt for _, cnt in top10]

            plt.figure(figsize=(8, 4))
            plt.bar(readable(chars), freqs, color='lightgreen')
            plt.xlabel('Characters')
            plt.ylabel('Frequency')
            plt.title(f"Top 10 Most Common Characters for Dataset: {name}")
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            out_plot = os.path.join(output_dir, f"{name}_top10.png")
            plt.savefig(out_plot, dpi=150)
            plt.show()
            print(f"✓ Saved plot to {out_plot}")

    # Save global summary
    with open(os.path.join(output_dir, "ALL_DATASETS_SUMMARY.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)

    print("\n🎉 All dataset analyses complete!")
    return summary

def get_comparisons():
    with open("dataset_reports/ALL_DATASETS_SUMMARY.json", "r", encoding="utf-8") as f:
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
