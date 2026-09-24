from citecheck.segment import parse_group, split


def test_parse_group():
    assert parse_group("1") == [1]
    assert parse_group("1, 3-5") == [1, 3, 4, 5]
    assert parse_group("2–3") == [2, 3]
    assert parse_group("1-999") == []  # absurd range ignored


def test_split_basic_and_trailing_citations():
    s = split("Paris is the capital of France.[1] It has 2.1 million people [2][3]. Dr. Smith agrees, e.g. in 2020.")
    assert [x.text for x in s] == [
        "Paris is the capital of France.",
        "It has 2.1 million people.",
        "Dr. Smith agrees, e.g. in 2020.",
    ]
    assert [x.citations for x in s] == [[1], [2, 3], []]


def test_split_grouped_and_duplicate_citations():
    s = split("A is true [1, 2] and B [2]. C is also true[3-4]!")
    assert s[0].citations == [1, 2] and s[0].text == "A is true and B."
    assert s[1].citations == [3, 4] and s[1].text == "C is also true!"


def test_split_edge_cases():
    assert split("") == []
    assert split("   ") == []
    one = split("No terminal punctuation [1]")
    assert len(one) == 1 and one[0].citations == [1] and one[0].text == "No terminal punctuation"
    q = split('He said "stop." Then left.')
    assert [x.text for x in q] == ['He said "stop."', "Then left."]
    assert split("Version 3.5 shipped. Next.")[0].text == "Version 3.5 shipped."
