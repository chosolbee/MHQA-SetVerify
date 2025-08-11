def convert_chat_to_text(chat, tokenizer, use_docs_only=False):
    content = ""
    for message in chat:
        if message['role'] == 'user':
            content = message['content']
            break

    lines = content.split('\n')
    question = ""
    for line in lines:
        if line.startswith("Main question: "):
            question = line[len("Main question: "):]
            break

    sep_token = tokenizer.sep_token if tokenizer.sep_token else "[SEP]"

    if use_docs_only:  # docs only
        documents = []
        for line in lines:
            if line.startswith("Document: "):
                documents.append(line[len("Document: "):])

        text_parts = [question] + documents
        return sep_token.join(text_parts)
    else:  # full trace
        text_parts = [question]

        for line in lines:
            line = line.strip()
            if line.startswith("Follow up: "):
                text_parts.append(line)
            elif line.startswith("Document: "):
                text_parts.append(line)
            elif line.startswith("Intermediate answer: "):
                text_parts.append(line)

        return sep_token.join(text_parts)
