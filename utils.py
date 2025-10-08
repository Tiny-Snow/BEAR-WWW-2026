def get_prompt(dataset_name):
    if "Toy" in dataset_name:
        instruction_prompt = "Given a list of toys the user has played before, please recommend a new toy that the user likes to the user."
        history_prompt = "The user has played the following toys before: "
    elif "Book" in dataset_name:
        instruction_prompt = "Given a list of books the user has read before, please recommend a new book that the user likes to the user."
        history_prompt = "The user has read the following books before: "
    elif "Clothing" in dataset_name:
        instruction_prompt = "Given a list of clothing the user has worn before, please recommend a new clothing that the user likes to the user."
        history_prompt = "The user has worn the following clothing before: "
    elif "Office" in dataset_name:
        instruction_prompt = "Given a list of office products the user has used before, please recommend a new office product that the user likes to the user."
        history_prompt = "The user has used the following office products before: "
    elif "Beauty" in dataset_name:
        instruction_prompt = "Given a list of beauty products the user has used before, please recommend a new beauty product that the user likes to the user."
        history_prompt = "The user has used the following beauty products before: "
    elif "Electronic" in dataset_name:
        instruction_prompt = "Given a list of electronic products the user has used before, please recommend a new electronic product that the user likes to the user."
        history_prompt = "The user has used the following electronic products before: "
    elif "Game" in dataset_name:
        instruction_prompt = "Given a list of video games the user has played before, please recommend a new video game that the user likes to the user."
        history_prompt = "The user has played the following video games before: "
    elif "Music" in dataset_name:
        instruction_prompt = "Given a list of music the user has listened to before, please recommend a new music that the user likes to the user."
        history_prompt = "The user has listened to the following music before: "

    return instruction_prompt, history_prompt
