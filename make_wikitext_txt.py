from datasets import load_from_disk

dataset = load_from_disk(".\data\wikitext_test")
print(dataset)
with open(f"wikitext-2-raw-test.txt", "w", encoding="utf-8") as f:
    for line in dataset["text"]:
        # 空行も残す（wikitext-raw の特徴）
        f.write(line)
        print(line)
