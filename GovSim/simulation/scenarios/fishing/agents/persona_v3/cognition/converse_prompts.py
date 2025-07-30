from datetime import datetime
from typing import List, Tuple

import numpy as np

from simulation.persona.common import PersonaIdentity
from simulation.utils import ModelWandbWrapper
from pathfinder import assistant, system, user

from .utils import (
    consider_identity_persona_prompt,
    conversation_to_string_with_dash,
    get_sytem_prompt,
    list_to_comma_string,
    list_to_string_with_dash,
    location_time_info,
    memory_prompt,
    reasoning_steps_prompt,
)

def get_input_with_validation(prompt: str, valid_answers: List[str] = None) -> str:
    """
    Prompts the user for input and validates it against a list of acceptable answers if provided.

    Parameters:
    - prompt (str): The question or prompt to display to the user.
    - valid_answers (List[str], optional): A list of valid answers. If None, any non-empty input is accepted.

    Returns:
    - str: The validated user input.
    """
    while True:
        user_input = input(prompt).strip()
        if user_input:
            if valid_answers is None or user_input.lower() in valid_answers:
                return user_input
            else:
                print(f"Invalid input. Please enter one of: {', '.join(valid_answers)}.")
        else:
            print("Input cannot be empty. Please try again.")


def get_human_answer_converse(prompt: str, target_personas: list[PersonaIdentity]) -> Tuple[str, str, str]:
    """
    Collects a response, whether the utterance has ended, and the next speaker from the human.

    Parameters:
    - prompt (str): The question or prompt to display to the human.
    - target_personas (List[str]): List of possible speakers to choose from.

    Returns:
    - Tuple[str, str, str]: A tuple containing the utterance, whether the utterance ended ("yes" or "no"),
      and the next speaker.
    """
    # Get the human's utterance
    utterance = get_input_with_validation(prompt)

    # Get whether the utterance has ended
    utterance_ended = get_input_with_validation(
        "Utterance ended (Yes, No): ", valid_answers=["yes", "no"]
    )

    # Get the next speaker
    speaker_prompt = f"Next speaker ({list_to_comma_string([t.name for t in target_personas])}): "
    next_speaker = get_input_with_validation(speaker_prompt)

    return utterance, utterance_ended, next_speaker


def prompt_converse_utterance_in_group(
    model: ModelWandbWrapper,
    init_persona: PersonaIdentity,
    target_personas: list[PersonaIdentity],
    init_retrieved_memory: list[str],
    current_location: str,
    current_time: datetime,
    current_context: str,
    current_conversation: list[tuple[str, str]],
) -> tuple[str, bool, str]:
    lm = model.start_chain(
        init_persona.name, "cognition_converse", "converse_utterance"
    )

    with user():
        lm += f"{get_sytem_prompt(init_persona)}\n"
        lm += location_time_info(current_location, current_time)
        # List key memories of the initial persona
        lm += memory_prompt(init_persona, init_retrieved_memory)
        # Provide the current context
        lm += "\n"
        # lm += f"Current context: {current_context}\n\n"
        # Describe the group chat scenario
        lm += (
            f"Scenario: {list_to_comma_string([t.name for t in target_personas])} are "
            "engaged in a group chat."
        )
        lm += "\nConversation so far:\n"
        lm += f"{conversation_to_string_with_dash(current_conversation)}\n\n"
        # Define the task for the language model
        lm += (
            f"Task: What would you say next in the group chat? "
            "Ensure the conversation flows naturally and avoids repetition. "
            "Determine if your response concludes the conversation. "
            "If not, identify the next speaker.\n\n"
        )
        # Define the format for the output
        REPONSE = "Response: "
        ANSWER_STOP = "Conversation conclusion by me: "
        NEXT_SPEAKER = "Next speaker: "

        lm += "Output format:\n"
        lm += REPONSE + "[fill in]\n"
        lm += ANSWER_STOP + "[yes/no]\n"
        lm += NEXT_SPEAKER + "[fill in]\n"
    with assistant():
        lm += REPONSE
        if False and init_persona.type == "human_agent": # Turn of human interaction for testing
            start_time_ms = datetime.now().timestamp() * 1000
            current_prompt = lm._current_prompt()

            utterance, utterance_ended_text, next_speaker = get_human_answer_converse(current_prompt, target_personas)
            print(f"\n\n\nreponses from human_converse: {utterance}, {utterance_ended_text}, {next_speaker}\n\n\n")
            lm += utterance
            lm = lm.set("utterance", utterance)
            lm = lm.set("utterance_ended", utterance_ended_text.lower())
            lm = lm.set("next_speaker", next_speaker)

            utterance_ended = (utterance_ended_text.lower() == "yes")
            model.log_human_output(start_time_ms, current_prompt, lm, name="utterance", default_value="yes")
        else:
            lm = model.gen(
                lm,
                name="utterance",
                default_value="",
                stop_regex=r"Conversation conclusion by me:",  # name can be mispelled by LLM sometimes
            )
            utterance = lm["utterance"].strip()
            if len(utterance) > 0 and utterance[-1] == '"' and utterance[0] == '"':
                    utterance = utterance[1:-1]
            lm += ANSWER_STOP
            lm = model.select(
                lm,
                name="utterance_ended",
                options=["yes", "no", "No", "Yes"],
                default_value="yes",
            )
            utterance_ended = lm["utterance_ended"].lower() == "yes"

            if utterance_ended:
                next_speaker = None
            else:
                lm += "\n"
                lm += NEXT_SPEAKER
                options = [t.name for t in target_personas]
                lm = model.select(
                    lm,
                    name="next_speaker",
                    options=options,
                    default_value=options[0],
                )
                assert lm["next_speaker"] in options
                next_speaker = lm["next_speaker"]

    model.end_chain(init_persona.name, lm)
    return utterance, utterance_ended, next_speaker, lm.html()


def prompt_summarize_conversation_in_one_sentence(
    model: ModelWandbWrapper,
    conversation: list[tuple[str, str]],
):
    lm = model.start_chain(
        "framework",
        "cognition_converse",
        "prompt_summarize_conversation_in_one_sentence",
    )

    with user():
        lm += f"Conversation:\n"
        lm += f"{conversation_to_string_with_dash(conversation)}\n\n"
        lm += "Summarize the conversation above in one sentence."
    with assistant():
        lm = model.gen(lm, name="summary", default_value="", stop_regex=r"\.")
        summary = lm["summary"] + "."

    model.end_chain("framework", lm)
    return summary, lm.html()
