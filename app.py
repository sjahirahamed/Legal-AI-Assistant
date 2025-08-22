from retriever import LawRetriever

if __name__ == "__main__":
    retriever = LawRetriever(data_path="data/laws.json")

    # Example crime scene description
    scene_description = (
        "A man intentionally stabbed another person with a knife during an argument, "
        "causing severe injuries but not death."
    )

    # Step 1: Retrieve relevant IPC sections
    matched_laws = retriever.find_relevant_laws(scene_description)

    print("🔎 Relevant IPC Sections Found:")
    for law in matched_laws:
        print(f"- {law['section']}: {law['title']}")

    # Step 2: Generate legal explanation with Gemini
    explanation = retriever.make_legal_explanation(scene_description, matched_laws)

    print("\n📖 Legal Explanation:\n")
    print(explanation)
