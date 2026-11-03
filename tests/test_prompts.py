from conftest import make

from finqa_ft.prompts import extract_program, messages, target


def test_extract_program():
    out = "<think>add(1, 2)</think>\nreasoning\nProgram: subtract(5, 3)"
    assert extract_program(out) == "subtract(5, 3)"
    assert extract_program("Program: add(1, 2)\nProgram: `divide(#0, 2)`") == "divide(#0, 2)"
    assert extract_program("the answer:\ndivide(1, 2)\nthanks") == "divide(1, 2)"
    assert extract_program("<think>never closed add(1, 2)") is None
    assert extract_program("no program here") is None
    assert extract_program("Program:\n`add(1, 2)`") == "add(1, 2)"
    assert extract_program("text\nProgram:") is None


def test_messages_same_input_with_and_without_shots(ex):
    shot = make("B/2/p.pdf-1", "q?", "add(1, 2)")
    zero, few = messages(ex), messages(ex, [shot])
    assert zero[0] == few[0]
    assert few[1]["content"].endswith(zero[1]["content"])
    assert "Program: add(1, 2)" in few[1]["content"]
    assert target(ex) == "Program: subtract(5, 3)"
